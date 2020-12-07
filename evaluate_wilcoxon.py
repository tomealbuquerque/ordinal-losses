import argparse
parser = argparse.ArgumentParser()
parser.add_argument('metric')
parser.add_argument('K', type=int)
parser.add_argument('--outputs1', nargs='+', required=True)
parser.add_argument('--outputs2', nargs='+', required=True)
args = parser.parse_args()
assert len(args.outputs1) == len(args.outputs2), (len(args.outputs1), len(args.outputs2))

from scipy.stats import wilcoxon
import mydataset, mymetrics
import pickle
import numpy as np

YY = pickle.load(open(f'data/k{args.K}.pickle', 'rb'))
res1 = []
res2 = []

for output1, output2 in zip(sorted(args.outputs1), sorted(args.outputs2)):
    fields = output1[:-4].split('-')[1:-1]
    fields = {k: v for k, v in zip(fields[::2], fields[1::2])}
    fold = int(fields['fold'])
    Y = YY[fold]['test'][1]
    Yhat = mymetrics.to_classes(np.loadtxt(output1, delimiter=','), 'mode')
    res1.append(getattr(mymetrics, args.metric)(Y, Yhat))
    Yhat = mymetrics.to_classes(np.loadtxt(output2, delimiter=','), 'mode')
    res2.append(getattr(mymetrics, args.metric)(Y, Yhat))

print(wilcoxon(res1, res2))
