from torch.utils.data import Dataset
from torchvision import models, transforms
import pickle
import numpy as np

class MyDataset(Dataset):
    def __init__(self, type, transform, K, fold):
        self.X, self.Y = pickle.load(open(f'data/k{K}.pickle', 'rb'))[fold][type]
        self.transform = transform

    def __getitem__(self, i):
        X = self.transform(self.X[i])
        Y = self.Y[i]
        return X, Y

    def __len__(self):
        return len(self.X)

aug_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(180, (0, 0.1), (0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(saturation=(0.5, 2.0)),
    transforms.ToTensor(),  # vgg normalization
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),  # vgg normalization
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['train', 'test'])
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    ds = MyDataset(args.type, aug_transforms, 7, 0)
    X, Y = ds[0]
    print('X:', X.min(), X.max(), X.shape, X.dtype)
    print(type, np.bincount(ds.Y) / len(ds.Y))
    plt.imshow(np.transpose((X-X.min()) / (X.max()-X.min()), (1, 2, 0)))
    plt.show()
