# -*- coding: utf-8 -*-

from __future__ import division, print_function

"""
Script to train an RTE LSTM.
"""

import sys
import argparse
import tensorflow as tf

import ioutils
import utils
from classifiers import LSTMClassifier, MultiFeedForwardClassifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('embeddings',
                        help='Text or numpy file with word embeddings')
    parser.add_argument('train', help='JSONL or TSV file with training corpus')
    parser.add_argument('validation',
                        help='JSONL or TSV file with validation corpus')
    parser.add_argument('save', help='Directory to save the model files')
    parser.add_argument('model', help='Type of architecture',
                        choices=['lstm', 'mlp'])
    parser.add_argument('--vocab', help='Vocabulary file (only needed if numpy'
                                        'embedding file is given)')
    parser.add_argument('-e', dest='num_epochs', default=10, type=int,
                        help='Number of epochs')
    parser.add_argument('-b', dest='batch_size', default=32, help='Batch size',
                        type=int)
    parser.add_argument('-u', dest='num_units', help='Number of hidden units',
                        default=100, type=int)
    parser.add_argument('--no-proj', help='Do not project input embeddings to '
                                          'the same dimensionality used by '
                                          'internal networks',
                        action='store_false', dest='no_project')
    parser.add_argument('-d', dest='dropout', help='Dropout keep probability',
                        default=1.0, type=float)
    parser.add_argument('-c', dest='clip_norm', help='Norm to clip training '
                                                     'gradients',
                        default=None, type=float)
    parser.add_argument('-r', help='Learning rate', type=float, default=0.001,
                        dest='rate')
    parser.add_argument('--lang', choices=['en', 'pt'], default='en',
                        help='Language (default en; only affects tokenizer)')
    parser.add_argument('--lower', help='Lowercase the corpus (use it if the '
                                        'embedding model is lowercased)',
                        action='store_true')
    parser.add_argument('--use-intra', help='Use intra-sentence attention',
                        action='store_true', dest='use_intra')
    parser.add_argument('--l2', help='L2 normalization constant', type=float,
                        default=0.0)
    parser.add_argument('--report', help='Number of batches between '
                                         'performance reports',
                        default=100, type=int)
    parser.add_argument('-v', help='Verbose', action='store_true',
                        dest='verbose')
    parser.add_argument('--optim', help='Optimizer algorithm',
                        default='adagrad',
                        choices=['adagrad', 'adadelta', 'adam'])

    args = parser.parse_args()

    utils.config_logger(args.verbose)
    logger = utils.get_logger('train')
    logger.debug('Training with following options: %s' % ' '.join(sys.argv))
    train_pairs = ioutils.read_corpus(args.train, args.lower, args.lang)
    valid_pairs = ioutils.read_corpus(args.validation, args.lower, args.lang)

    # whether to generate embeddings for unknown, padding, null
    word_dict, embeddings = ioutils.load_embeddings(args.embeddings, args.vocab,
                                                    True, normalize=True)

    logger.info('Converting words to indices')
    # find out which labels are there in the data
    # (more flexible to different datasets)
    label_dict = utils.create_label_dict(train_pairs)
    train_data = utils.create_dataset(train_pairs, word_dict, label_dict)
    valid_data = utils.create_dataset(valid_pairs, word_dict, label_dict)

    ioutils.write_params(args.save, lowercase=args.lower, language=args.lang)
    ioutils.write_label_dict(label_dict, args.save)
    ioutils.write_extra_embeddings(embeddings, args.save)

    msg = '{} sentences have shape {} (firsts) and {} (seconds)'
    logger.debug(msg.format('Training',
                            train_data.sentences1.shape,
                            train_data.sentences2.shape))
    logger.debug(msg.format('Validation',
                            valid_data.sentences1.shape,
                            valid_data.sentences2.shape))

    sess = tf.InteractiveSession()
    logger.info('Creating model')
    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]

    if args.model == 'mlp':
        model = MultiFeedForwardClassifier(args.num_units, 3, vocab_size,
                                           embedding_size,
                                           use_intra_attention=args.use_intra,
                                           training=True,
                                           project_input=args.no_project,
                                           optimizer=args.optim)
    else:
        model = LSTMClassifier(args.num_units, 3, vocab_size,
                               embedding_size, training=True,
                               project_input=args.no_project)

    model.initialize(sess, embeddings)

    total_params = utils.count_parameters()
    logger.debug('Total parameters: %d' % total_params)

    logger.info('Starting training')
    model.train(sess, train_data, valid_data, args.save, args.rate,
                args.num_epochs, args.batch_size, args.dropout, args.l2,
                args.clip_norm, args.report)
