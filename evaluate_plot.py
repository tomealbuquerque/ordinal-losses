import argparse
parser = argparse.ArgumentParser()
parser.add_argument('trueK', type=int)
parser.add_argument('K', type=int)
parser.add_argument('architecture')
parser.add_argument('method')
parser.add_argument('outputs', nargs='+')
args = parser.parse_args()

import numpy as np
import pickle
import mydataset

fold = 0
Y = pickle.load(open(f'data/k{args.K}.pickle', 'rb'))[fold]['test'][1]
start = f'output-architecture-{args.architecture}-method-{args.method}-K-{args.K}-fold-{fold}-'
filename = [o for o in args.outputs if o.split('/')[-1].startswith(start)]
assert len(filename), f'Empty filenames starting with {start}'
Yhat = np.loadtxt(filename[0], delimiter=',')

print(r'\addplot coordinates {', end='')
for k in range(args.K):
    print('(%d, %.4f)' % (k+1, np.mean(Yhat[Y == (args.trueK-1)][:, k])), end=' ')
print(r'};  %',  args.method, args.trueK)
