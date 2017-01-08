# -*- coding: utf-8 -*-

from __future__ import print_function

"""
Interactive evaluation for the RTE networks.
"""

import argparse
import tensorflow as tf
import numpy as np

from classifiers import multimlp
import utils
import ioutils


class SentenceWrapper(object):
    """
    Class for the basic sentence preprocessing needed to make it readable
    by the networks.
    """
    def __init__(self, sentence, word_dict, lowercase, language='en'):
        self.sentence = sentence
        tokenize = utils.get_tokenizer(language)
        if lowercase:
            pre_tokenize = sentence.lower()
        else:
            pre_tokenize = sentence
        self.tokens = tokenize(pre_tokenize)
        self.indices = [word_dict[token] for token in self.tokens_with_null]
        self.padding_index = word_dict[utils.PADDING]

    def __len__(self):
        return len(self.tokens)

    @property
    def tokens_with_null(self):
        return [utils.GO] + self.tokens

    def convert_sentence(self):
        """
        Convert a sequence of tokens into the input array used by the network
        :return: the vector to be given to the network
        """
        indices = np.array(self.indices)
        # padded = np.pad(indices, (0, num_time_steps - len(indices)),
        #                 'constant', constant_values=self.padding_index)
        return indices.reshape((1, -1))


def print_attention(tokens1, tokens2, attention):
    """
    Print the attention from tokens1 over tokens2
    """
    # multiply by 10 to make it easier to visualize
    attention_bigger = attention * 10
    max_length_sent1 = max([len(t) for t in tokens1])

    # create formatting string to match the size of the tokens
    att_formatters = ['{:>%d.2f}' % len(t) for t in tokens2]

    # first line has whitespace in the first sentence column and
    # then the second one
    blank = ' ' * max_length_sent1

    # take at least length 4 to fit the 9.99 format
    formatted_sent2 = ['{:>4}'.format(token) for token in tokens2]
    first_line = blank + '\t' + '\t'.join(formatted_sent2)
    print(first_line)

    for token, att in zip(tokens1, attention_bigger):

        values = [fmt.format(x)
                  for x, fmt in zip(att, att_formatters)]
        fmt_str = '{:>%d}' % max_length_sent1
        formatted_token = fmt_str.format(token)
        line = formatted_token + '\t' + '\t'.join(values)
        print (line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('load', help='Directory with saved model files')
    parser.add_argument('embeddings', help='Text or numpy file with word embeddings')
    parser.add_argument('--vocab', help='Vocabulary file (only needed if numpy'
                                        'embedding file is given)')
    parser.add_argument('-a', help='Print attention values', dest='attention',
                        action='store_true')
    parser.add_argument('-i', help='Run inference classifier', dest='inference',
                        action='store_true')
    args = parser.parse_args()

    utils.config_logger(verbose=False)
    logger = utils.get_logger()
    params = ioutils.load_params(args.load)
    if args.inference:
        label_dict = ioutils.load_label_dict(args.load)
        number_to_label = {v: k for (k, v) in label_dict.items()}

    logger.info('Reading model')
    sess = tf.InteractiveSession()
    model = multimlp.MultiFeedForwardClassifier.load(args.load, sess)
    word_dict, embeddings = ioutils.load_embeddings(args.embeddings, args.vocab,
                                                    generate=False,
                                                    load_extra_from=args.load,
                                                    normalize=True)
    model.initialize_embeddings(sess, embeddings)

    ops = []
    if args.inference:
        ops.append(model.answer)
    if args.attention:
        ops.append(model.inter_att1)
        ops.append(model.inter_att2)

    while True:
        sent1 = raw_input('Type sentence 1: ').decode('utf-8')
        sent2 = raw_input('Type sentence 2: ').decode('utf-8')
        sent1 = SentenceWrapper(sent1, word_dict,
                                params['lowercase'], params['language'])
        sent2 = SentenceWrapper(sent2, word_dict,
                                params['lowercase'], params['language'])

        vector1 = sent1.convert_sentence()
        vector2 = sent2.convert_sentence()
        # +1 for NULL symbol
        size1 = len(sent1) + 1
        size2 = len(sent2) + 1

        feeds = {model.sentence1: vector1,
                 model.sentence2: vector2,
                 model.sentence1_size: [size1],
                 model.sentence2_size: [size2],
                 model.dropout_keep: 1.0}

        results = sess.run(ops, feed_dict=feeds)
        if args.inference:
            answer = results.pop(0)
            print('Model answer:', number_to_label[answer[0]])

        if args.attention:
            att1 = results.pop(0)
            att2 = results.pop(0)
            print('Attention sentence 1:')
            print_attention(sent1.tokens_with_null,
                            sent2.tokens_with_null, att1[0])
            print('Attention sentence 2:')
            print_attention(sent2.tokens_with_null,
                            sent1.tokens_with_null, att2[0])

        print()
