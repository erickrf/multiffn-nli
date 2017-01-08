# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import argparse
import tensorflow as tf

import utils
import ioutils
import classifiers

"""
Evaluate the performance of an NLI model on a dataset
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Directory with saved model')
    parser.add_argument('dataset', help='JSONL or TSV file with data to evaluate on')
    parser.add_argument('embeddings', help='Numpy embeddings file')
    parser.add_argument('vocabulary', help='Text file with embeddings vocabulary')
    parser.add_argument('-v', help='Verbose', action='store_true', dest='verbose')
    args = parser.parse_args()

    utils.config_logger(verbose=args.verbose)

    sess = tf.InteractiveSession()
    model = classifiers.MultiFeedForwardClassifier.load(args.model, sess)
    word_dict, embeddings = ioutils.load_embeddings(args.embeddings,
                                                    args.vocabulary,
                                                    generate=False,
                                                    load_extra_from=args.model,
                                                    normalize=True)
    model.initialize_embeddings(sess, embeddings)
    label_dict = ioutils.load_label_dict(args.model)
    params = ioutils.load_params(args.model)

    pairs = ioutils.read_corpus(args.dataset, params['lowercase'],
                                params['language'])
    dataset = utils.create_dataset(pairs, word_dict, label_dict)
    loss, acc = model.evaluate(sess, dataset)
    print('Loss: %f' % loss)
    print('Accuracy: %f' % acc)
