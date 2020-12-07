import argparse
parser = argparse.ArgumentParser()
parser.add_argument('K', type=int)
parser.add_argument('architecture')
parser.add_argument('method')
parser.add_argument('toclass')
parser.add_argument('outputs', nargs='+')
args = parser.parse_args()

import numpy as np
import pickle
import mydataset, mymetrics
from sklearn.metrics import confusion_matrix

YY = [pickle.load(open(f'data/k{args.K}.pickle', 'rb'))[fold]['test'][1] for fold in range(10)]
conf_matrices = mymetrics.evaluate(YY, args.K, args.outputs, False, args.architecture, args.method, lambda y, yhat: confusion_matrix(y, yhat, normalize='true'), args.toclass)

conf_matrix = (100*np.mean(conf_matrices, 0)).astype(int)

print(r'''\documentclass{standalone}
\usepackage{graphicx}
\usepackage{colortbl}
\usepackage{multirow}
\begin{document}
\begin{tabular}{cc|''', end='')
for _ in range(args.K):
    print('c', end='')
print(r'''}
& \multicolumn{1}{c}{} & \multicolumn{''' + str(args.K) + r'''}{c}{Predicted} \\''')
print(r'\parbox[t]{2mm}{\multirow{' + str(args.K) + r'}{*}{\rotatebox{90}{Actual}}} &', end='')
for k in range(args.K):
    print(' & ' + str(k+1))
print(r'\\')
for k, line in enumerate(conf_matrix):
    print(' & ' + str(k+1) + ' & ' + ' & '.join('\\cellcolor[rgb]{%.2f,%.2f,1}\\textcolor{%s}{%d}' % (1-v/100, 1-v/100, 'white' if v >= 50 else 'black', v) for v in line) + r'\\')
print(r'''\end{tabular}
\end{document}
''')
