import argparse
parser = argparse.ArgumentParser()
parser.add_argument('datapath')
parser.add_argument('K', choices=[2, 4, 7], type=int)
args = parser.parse_args()

from sklearn.model_selection import StratifiedKFold
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import pickle
import os

classes_to_y = {
    2: [0]*3 + [1]*4,
    4: [0]*3 + [1] + [2]*2 + [3],
    7: list(range(7)),
}

# load images
X = []
Y = []
for c in range(1, 8):
    dirname = f'{args.datapath}/class_{c}'
    imgs = [imread(os.path.join(dirname, f)) for f in os.listdir(dirname)]
    imgs = [resize(img, (224, 224))*255 for img in imgs]
    X += imgs
    Y += [classes_to_y[args.K][c-1]] * len(imgs)

X = np.array(X, np.uint8)
Y = np.array(Y, np.uint8)

# kfold
state = np.random.RandomState(1234)
kfold = StratifiedKFold(10, shuffle=True, random_state=state)
folds = [{'train': (X[tr], Y[tr]), 'test': (X[ts], Y[ts])} 
    for tr, ts in kfold.split(X, Y)]
pickle.dump(folds, open(f'data/k{args.K}.pickle', 'wb'))
