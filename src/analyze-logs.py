# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import argparse
from collections import namedtuple
import re

"""
Extract the top accuracies or smallest losses from every training log file
in a directory.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', help='Log files',
                        nargs='+')
    parser.add_argument('--metric', help='What metric to sort by',
                        choices=['accuracy', 'loss'], default='accuracy')
    args = parser.parse_args()

    regex_acc = r'Validation accuracy: (\d+\.\d+)'
    regex_loss = r'Validation loss: (\d+\.\d+)'
    Performance = namedtuple('Performance', ['acc', 'loss'])
    performances = {}

    for filename in args.files:
        with open(filename, 'rb') as f:
            text = f.read()

        accs = [float(val) for val in re.findall(regex_acc, text)]
        max_acc = max(accs)
        losses = [float(val) for val in re.findall(regex_acc, text)]
        best_loss = min(losses)
        performances[filename] = Performance(max_acc, best_loss)

    if args.metric == 'accuracy':
        sort_fn = lambda k: performances[k].acc
    else:
        sort_fn = lambda k: performances[k].loss

    sorted_names = sorted(performances, key=sort_fn, reverse=True)
    for name in sorted_names:
        print(name, '\t', performances[name].acc, '\t', performances[name].loss)
