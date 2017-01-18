# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import os
import argparse
import tensorflow as tf
import matplotlib
matplotlib.use('TKAgg')  # necessary on OS X
from matplotlib import pyplot as pl

from classifiers import MultiFeedForwardClassifier

"""
Plot weights in a RTE FF model
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Directory containing the '
                                      'trained model')
    parser.add_argument('-o', help='Output directory for images '
                                   '(default is same as the model)',
                        dest='output')
    args = parser.parse_args()
    output = args.output or args.model

    sess = tf.InteractiveSession()
    model = MultiFeedForwardClassifier.load(args.model, sess)

    for var in tf.trainable_variables():
        if 'bias' in var.name:
            continue

        # create valid filenames
        name = var.name.replace(':0', '').replace('/', '_')
        values = var.eval()
        pl.matshow(values)
        pl.colorbar()
        path = os.path.join(output, name)
        pl.savefig(path)
