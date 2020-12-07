from sklearn import metrics
from scipy import stats
from wilson_metric import wilson_index
import numpy as np

def to_classes(p, method):
    if method == 'mode':
        return p.argmax(1)
    if method == 'mean':  # so-called expectation trick
        kk = np.arange(p.shape[1])
        return np.round(np.sum(p * kk, 1)).astype(int)
    if method == 'median':
        # the weighted median is the value whose cumulative probability is 0.5
        pc = np.cumsum(p, 1)
        return np.sum(pc < 0.5, 1)

def acc(y, yhat):
    return metrics.accuracy_score(y, yhat)

def bacc(y, yhat):
    return metrics.balanced_accuracy_score(y, yhat)

def mae(y, yhat):
    return metrics.mean_absolute_error(y, yhat)

def amae(y, yhat):
    K = y.max()+1
    mae_k = [np.mean(np.abs(y[y == k] - yhat[y == k])) for k in range(K)]
    return np.mean(mae_k)

def f1(y, yhat):
    return metrics.f1_score(y, yhat, average='macro')

def wilson(y, yhat):
    return wilson_index(y, yhat)

def roc_auc(y, yhat):
    yhat = yhat / yhat.sum(1, keepdims=True)
    return metrics.roc_auc_score(y, yhat, multi_class='ovo')

def tau(y, yhat):
    return stats.kendalltau(y, yhat)[0]

def gini(y, yhat):
    array = yhat
    # https://github.com/oliviaguest/gini
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    ret = ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))
    return ret


properties = {
    'acc': ('max', 100, 1, False),
    'bacc': ('max', 100, 1, False),
    'mae': ('min', 1, 2, False),
    'amae': ('min', 1, 2, False),
    'f1': ('max', 100, 1, False),
    'wilson': ('min', 100, 1, False),
    'roc_auc': ('max', 100, 1, True),
    'tau': ('max', 100, 1, False),
    'gini': (None, 100, 1, True),
}

def evaluate(YY, K, outputs, proba, architecture, method, metric, toclass):
    scores = []
    for fold in range(10):
        Y = YY[fold]
        start = f'output-architecture-{architecture}-method-{method}-K-{K}-fold-{fold}-'
        filename = [o for o in outputs if o.split('/')[-1].startswith(start)]
        assert len(filename), f'Empty filenames starting with {start}'
        Yhat = np.loadtxt(filename[0], delimiter=',')
        if not proba:  # need to convert to classes
            Yhat = to_classes(Yhat, toclass)
        scores.append(metric(Y, Yhat))
    return scores
