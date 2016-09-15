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
    args = parser.parse_args()

    logger = utils.get_logger()

    logger.info('Reading model')
    sess = tf.InteractiveSession()
    model = multimlp.MultiFeedForward.load(args.load, sess)
    word_dict = readdata.read_word_dict(args.load)

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
