# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import argparse
import logging
import tensorflow as tf

import utils
import ioutils
import classifiers

"""
Script to pretrain a decomposable text entailment to align words
between two sentences. The training objective is to achieve a
naïve alignment in which the alignment score is equally divided
between synonyms of a word, or all set to null, when there is no
synonym.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', help='File with aligned training corpus')
    parser.add_argument('valid', help='File with aligned validation corpus')
    parser.add_argument('save', help='Directory to save the model files')
    parser.add_argument('embeddings', help='Numpy file with word embeddings')
    parser.add_argument('vocabulary', help='Text file with embedding vocabulary')
    parser.add_argument('--lower', action='store_true', dest='lower',
                        help='Convert corpus to lowercase (use this if '
                             'the embedding model is lower cased)')
    parser.add_argument('-e', dest='num_epochs', default=10, type=int,
                        help='Number of epochs')
    parser.add_argument('-b', dest='batch_size', default=32, help='Batch size',
                        type=int)
    parser.add_argument('-u', dest='num_units', help='Number of hidden units',
                        default=100, type=int)
    parser.add_argument('-d', dest='dropout', help='Dropout keep probability',
                        default=1.0, type=float)
    parser.add_argument('-c', dest='clip_norm', help='Norm to clip training gradients',
                        default=None, type=float)
    parser.add_argument('-r', help='Learning rate', type=float, default=0.001,
                        dest='rate')
    parser.add_argument('--l2', help='L2 normalization constant', type=float,
                        default=0.0)
    parser.add_argument('-v', help='Verbose', action='store_true', dest='verbose')
    parser.add_argument('--report', help='Number of batches between performance reports',
                        default=100, type=int)

    args = parser.parse_args()

    utils.config_logger(args.verbose)
    word_dict, embeddings = ioutils.load_embeddings(args.embeddings, args.vocabulary)
    train_dataset = utils.create_alignment_dataset(args.train, args.lower, word_dict)
    valid_dataset = utils.create_alignment_dataset(args.valid, args.lower, word_dict)

    model = classifiers.MultiFeedForwardClassifier(args.num_units, 3,
                                                   embeddings.shape[0],
                                                   embeddings.shape[1], False,
                                                   training=False)
    sess = tf.InteractiveSession()
    model.initialize(sess, embeddings)
    pretrainer = classifiers.AlignPretrainer(model)
    pretrainer.initialize(sess)

    total_params = utils.count_parameters()
    logging.debug('Total parameters: %d' % total_params)

    logging.debug('Starting training')
    pretrainer.train(sess, train_dataset, valid_dataset, args.save,
                     args.rate, args.num_epochs, args.batch_size, args.dropout,
                     args.l2,  args.clip_norm, args.report)
