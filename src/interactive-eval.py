# -*- coding: utf-8 -*-

from __future__ import print_function

"""
Interactive evaluation for the RTE LSTM.
"""

import argparse
import tensorflow as tf
import numpy as np

import multimlp
import utils
import readdata


def convert_tokens(sentence, word_dict, num_time_steps, prepend=None):
    """
    Convert a sequence of tokens into the input array used by the network
    :param prepend: if not None, prepend this value to the sequence
    """
    values = [word_dict[token] for token in sentence]
    if prepend is not None:
        values = [prepend] + values

    indices = np.array(values)
    padded = np.pad(indices, (0, num_time_steps - len(indices)),
                    'constant', constant_values=word_dict[utils.PADDING])
    return padded.reshape((num_time_steps, 1))


def print_attention(tokens, attentions):
    attentions *= 10
    max_len = max([len(t) for t in tokens])
    fmt_str = '{{:{}}} {{}}'.format(max_len)

    for token, att in zip(tokens, attentions):
        values = ["{:0.2f}".format(x) for x in att]
        print (fmt_str.format(token, values))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('load', help='Directory with saved model files')
    parser.add_argument('embeddings', help='Text or numpy file with word embeddings')
    parser.add_argument('--vocab', help='Vocabulary file (only needed if numpy'
                                        'embedding file is given)')
    args = parser.parse_args()

    utils.config_logger(verbose=False)
    logger = utils.get_logger()

    logger.info('Reading model')
    sess = tf.InteractiveSession()
    model = multimlp.MultiFeedForward.load(args.load, sess)
    word_dict, embeddings = readdata.load_embeddings(args.embeddings, args.vocab,
                                                     generate=False,
                                                     load_extra_from=args.load)
    embeddings = utils.normalize_embeddings(embeddings)
    model.initialize_embeddings(sess, embeddings)
    number_to_label = {v: k for (k, v) in utils.label_map.items()}

    while True:
        sent1 = raw_input('Type sentence 1: ')
        sent2 = raw_input('Type sentence 2: ')
        tokens1 = utils.tokenize(sent1)
        tokens2 = utils.tokenize(sent2)
        vector1 = convert_tokens(tokens1, word_dict, model.max_time_steps1)
        vector2 = convert_tokens(tokens2, word_dict, model.max_time_steps2,
                                 prepend=word_dict[utils.GO])

        feeds = {model.sentence1: vector1,
                 model.sentence2: vector2,
                 model.sentence1_size: [len(tokens1)],
                 model.sentence2_size: [len(tokens2)+1],
                 model.dropout_keep: 1.0}

        answer = sess.run(model.answer, feed_dict=feeds)
        print('Model answer:', number_to_label[answer[0]])

        print()
