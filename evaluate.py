import argparse
parser = argparse.ArgumentParser()
parser.add_argument('metric')
parser.add_argument('K', type=int)
parser.add_argument('baseline')
parser.add_argument('outputs', nargs='+')
parser.add_argument('--architectures', nargs='+')
parser.add_argument('--methods', nargs='+')
parser.add_argument('--toclasses', nargs='+', choices=['mode', 'mean', 'median'])
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

metric = getattr(mymetrics, args.metric)
order, magnitude, places, proba = mymetrics.properties[args.metric]

architectures = args.architectures if args.architectures else sorted({os.path.basename(out).split('-')[2] for out in args.outputs})
methods = args.methods if args.methods else sorted({os.path.basename(out).split('-')[4] for out in args.outputs})

fields = args.outputs[0][:-4].split('-')[1:-1]
fields = {k: v for k, v in zip(fields[::2], fields[1::2])}

print(r'''\documentclass{standalone}
\begin{document}''')
print('\\begin{tabular}{l' + ('r'*len(methods)) + '}')
headers = args.headers if args.headers else methods
print('& ' + ' & '.join([r'\multicolumn{1}{c}{' + escape(h) + '}' for h in headers]) + '\\\\\\hline')

sum_avgs = []
winners = [0] * len(methods)

for architecture in architectures:
    print('\\textbf{%s}' % escape(architecture), end=' & ')
    avgs = []
    devs = []
    scores_per_fold = []
    for i, method in enumerate(methods):
        toclass = args.toclasses[i] if args.toclasses else 'mode'
        scores = mymetrics.evaluate(YY, args.K, args.outputs, proba, architecture, method, metric, toclass)
        scores_per_fold.append(scores)
        avgs.append(np.mean(scores))
        devs.append(np.std(scores))
    best = getattr(np, 'nanarg' + order)(avgs)
    for i, (avg, dev) in enumerate(zip(avgs, devs)):
        print('$', end='')
        bold = i == best
        statistic_test = 0
        if methods[i] != args.baseline:
            # compare this method against the baseline using a statistical test
            j = methods.index(args.baseline)
            this_folds = scores_per_fold[i]
            best_folds = scores_per_fold[j]
            _, pvalue = stats.ttest_rel(this_folds, best_folds)
            statistic_test = (pvalue <= 0.2) * np.sign(np.mean(scores_per_fold[i]) - np.mean(scores_per_fold[j])) * (2*(order == 'max')-1)
            if np.isnan(statistic_test):
                statistic_test = 0
        if bold:
            print(r'\mathbf{', end='')
        print(f'{avg*magnitude:.{places}f} \pm {dev*magnitude:.{places}f}', end='')
        if bold:
            print('}', end='')
        print('$', end='')
        symbols = {0: r'\hphantom{$\circ$}', 1: r'$\circ$', -1: r'$\bullet$'}
        print(f' {symbols[statistic_test]}', end='')
        if i < len(avgs)-1:
            print(' & ', end='')
        winners[i] += int(bold)
    print(' \\\\')
    sum_avgs.append(avgs)
print('\\hline')
print('\\textbf{Avg} & ' + ' & '.join([r'\multicolumn{1}{c}{' + f'{a*magnitude:.{places}f}' + '}' for a in np.mean(sum_avgs, 0)]) + '\\\\')
print('\\textbf{Winners} & ' + ' & '.join(r'\multicolumn{1}{c}{' + str(w) + '}' for w in winners) + '\\\\')
print(r'''\end{tabular}
\end{document}''')
