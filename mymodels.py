from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# References
# Beckham model: http://proceedings.mlr.press/v70/beckham17a/beckham17a.pdf
# Ordinal encoder: https://arxiv.org/pdf/0704.1028.pdf

# Utilities

def fact(x):
    return torch.exp(torch.lgamma(x+1))

def log_fact(x):
    return torch.lgamma(x+1)

ce = nn.CrossEntropyLoss()

def entropy_loss(Yhat):
    # https://en.wikipedia.org/wiki/Entropy_(information_theory)
    P = F.softmax(Yhat, -1)
    logP = F.log_softmax(Yhat, -1)
    N = P.shape[0]
    return -torch.sum(P * logP) / N

def neighbor_loss(margin, Yhat, Y):
    margin = torch.tensor(margin, device=device)
    P = F.softmax(Yhat, -1)
    K = P.shape[1]
    loss = 0
    for k in range(K-1):
        # force previous probability to be superior to next
        reg_gt = (Y >= k+1).float() * F.relu(margin+P[:,  k]-P[:, k+1])
        reg_lt = (Y <= k).float() * F.relu(margin+P[:, k+1]-P[:, k])
        loss += torch.mean(reg_gt + reg_lt)
    return loss

# Models

class Base(nn.Module):
    def __init__(self, pretrained_model, n_outputs):
        super().__init__()
        self.n_outputs = n_outputs
        model = getattr(models, pretrained_model)(pretrained=True)
        model = nn.Sequential(*tuple(model.children())[:-1])
        last_dimension = torch.flatten(model(torch.randn(1, 3, 224, 224))).shape[0]
        self.model = nn.Sequential(
            model,
            nn.Flatten(),
            nn.Dropout(0.2),
            # last_dimension is 512*7*7 for vgg16
            nn.Linear(last_dimension, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_outputs)
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, Yhat, Y):
        return ce(Yhat, Y)

    def to_proba(self, Yhat):
        return F.softmax(Yhat, 1)

    def to_classes(self, Phat, type):
        assert type in ('mode', 'mean', 'median')
        if type == 'mode':
            return Phat.argmax(1)
        if type == 'mean':  # so-called expectation trick
            kk = torch.arange(args.classes, device=device)
            return torch.round(torch.sum(Yhat * kk, 1)).long()
        if type == 'median':
            # the weighted median is the value whose cumulative probability is 0.5
            Pc = torch.cumsum(Phat, 1)
            return torch.sum(Pc < 0.5, 1)

class Beckham(Base):
    def forward(self, x):
        x = super().forward(x)
        x = nn.Softplus()(x)  # they apply softplus (relu) to avoid log(negative)
        K = self.n_outputs
        KK = torch.arange(1., K+1, device=device)
        return KK*torch.log(x) - x - log_fact(KK)

class OrdinalEncoder(Base):
    def __init__(self, pretrained_model, K):
        super().__init__(pretrained_model, K-1)
        self.K = K

    def loss(self, Yhat, Y):
        # if K=4, then
        #     Y=0 => Y_=[0, 0, 0]
        #     Y=1 => Y_=[1, 0, 0]
        #     Y=2 => Y_=[1, 1, 0]
        #     Y=3 => Y_=[1, 1, 1]
        KK = torch.arange(self.n_outputs, device=device).expand(Y.shape[0], -1)
        YY = (Y[:, None] > KK).float()
        return F.binary_cross_entropy_with_logits(Yhat, YY)

    def to_proba(self, Yhat):
        # we need to convert mass distribution into probabilities
        # i.e. P(Y >= k) into P(Y = k)
        # P(Y=0) = 1-P(Y≥1)
        # P(Y=1) = P(Y≥1)-P(Y≥2)
        # ...
        # P(Y=K-1) = P(Y≥K-1)
        Phat = torch.sigmoid(Yhat)
        Phat = torch.cat((1-Phat[:, :1], Phat[:, :-1] - Phat[:, 1:], Phat[:, -1:]), 1)
        return torch.clamp(Phat, 0, 1)

class Unimodal(Base):
    def __init__(self, pretrained_model, K):
        super().__init__(pretrained_model, 1)
        self.K = K

    def to_proba(self, Yhat):
        Phat = torch.sigmoid(Yhat)
        N = Yhat.shape[0]
        K = torch.tensor(self.K, dtype=torch.float, device=device)
        kk = torch.ones((N, self.K), device=device) * torch.arange(self.K, dtype=torch.float, device=device)[None]
        num = fact(K-1) * (Phat**kk) * (1-Phat)**(K-kk-1)
        den = fact(kk) * fact(K-kk-1)
        return num / den

    def to_log_proba(self, Yhat):
        log_Phat = F.logsigmoid(Yhat)
        log_inv_Phat = F.logsigmoid(-Yhat)
        N = Yhat.shape[0]
        K = torch.tensor(self.K, dtype=torch.float, device=device)
        kk = torch.ones((N, self.K), device=device) * torch.arange(self.K, dtype=torch.float, device=device)[None]
        num = log_fact(K-1) + kk*log_Phat + (K-kk-1)*log_inv_Phat
        den = log_fact(kk) + log_fact(K-kk-1)
        return num - den

class UnimodalCE(Unimodal):
    def loss(self, Yhat, Y):
        return F.nll_loss(self.to_log_proba(Yhat), Y)

class UnimodalMSE(Unimodal):
    def loss(self, Yhat, Y):
        Phat = self.to_proba(Yhat)
        Y_onehot = torch.zeros(Phat.shape[0], self.K, device=device)
        Y_onehot[range(Phat.shape[0]), Y] = 1
        return torch.mean((Phat - Y_onehot) ** 2)

class CO2(Base):
    def __init__(self, pretrained_model, K, lambda_, omega):
        super().__init__(pretrained_model, K)
        self.lambda_ = lambda_
        self.omega = omega

    def loss(self, Yhat, Y):
        return self.lambda_*ce(Yhat, Y) + neighbor_loss(self.omega, Yhat, Y)

class CO(CO2):
    def __init__(self, pretrained_model, K, lambda_, omega):
        super().__init__(pretrained_model, K, lambda_, 0)

class HO2(Base):
    def __init__(self, pretrained_model, K, lambda_, omega):
        super().__init__(pretrained_model, K)
        self.lambda_ = lambda_
        self.omega = omega

    def loss(self, Yhat, Y):
        return self.lambda_*entropy_loss(Yhat) + neighbor_loss(self.omega, Yhat, Y)
