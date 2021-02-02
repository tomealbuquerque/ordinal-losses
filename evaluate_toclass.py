import argparse
parser = argparse.ArgumentParser()
parser.add_argument('K', type=int)
parser.add_argument('outputs', nargs='+')
parser.add_argument('--metrics', nargs='+', required=True)
parser.add_argument('--architectures', nargs='+')
parser.add_argument('--methods', nargs='+')
parser.add_argument('--headers', nargs='+')
args = parser.parse_args()

import numpy as np
import pickle
import mydataset, mymetrics
from scipy import stats
import os

def escape(s):
    return s.replace('_', f'\_')

YY = [pickle.load(open(f'data/k{args.K}.pickle', 'rb'))[fold]['test'][1] for fold in range(10)]

architectures = args.architectures if args.architectures else sorted({os.path.basename(out).split('-')[2] for out in args.outputs})
methods = args.methods if args.methods else sorted({os.path.basename(out).split('-')[4] for out in args.outputs})

fields = args.outputs[0][:-4].split('-')[1:-1]
fields = {k: v for k, v in zip(fields[::2], fields[1::2])}

print(r'''\documentclass{standalone}
\begin{document}''')
print('\\begin{tabular}{l' + ('r'*len(methods)) + '}')
print('& ' + ' & '.join(map(escape, args.headers if args.headers else methods)) + '\\\\\\hline')

for toclass in ('mode', 'mean'):#, 'median'):
    print('\\multicolumn{%d}{c}{\\textbf{%s}}\\\\' % (len(methods)+1, toclass.title()))
    for metric in args.metrics:
        print('\\textbf{%s}' % escape(metric), end=' & ')
        order, magnitude, places, proba = mymetrics.properties[metric]
        metric = getattr(mymetrics, metric)
        avgs = []
        devs = []
        scores_per_fold = []
        for method in methods:
            scores = []
            for architecture in architectures:
                scores += mymetrics.evaluate(YY, args.K, args.outputs, proba, architecture, method, metric, toclass)
            scores_per_fold.append(scores)
            avgs.append(np.mean(scores))
            devs.append(np.std(scores))
        best = getattr(np, 'nanarg' + order)(avgs) if order else -1
        for i, (avg, dev) in enumerate(zip(avgs, devs)):
            print('$', end='')
            italic = bold = False
            if i == best:
                bold = True
            elif i >= 0:
                # compare this method against the best using a statistical test
                this_folds = scores_per_fold[i]
                best_folds = scores_per_fold[best]
                _, pvalue = stats.ttest_rel(this_folds, best_folds)
                italic = pvalue/2 > 0.10  # divide by two to make it one-sided
            if bold:
                print(r'\mathbf{', end='')
            if italic:
                print(r'\mathit{', end='')
            print(f'{avg*magnitude:.{places}f} \pm {dev*magnitude:.{places}f}', end='')
            if bold or italic:
                print('}', end='')
            print('$', end='')
            if i < len(avgs)-1:
                print(' & ', end='')
        print(' \\\\')
print('\\hline')
print(r'''\end{tabular}
\end{document}''')
