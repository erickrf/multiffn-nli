# -*- coding: utf-8 -*-

from __future__ import division

"""
Utility functions.
"""

import logging
import nltk
import os
import json
import numpy as np
from collections import Counter

tokenizer = nltk.tokenize.TreebankWordTokenizer()
label_map = {'neutral': 0,
             'entailment': 1,
             'contradiction': 2}
UNKNOWN = '**UNK**'
PADDING = '**PAD**'
GO = '**GO**'


class RTEDataset(object):
    """
    Class for better organizing a data set. It provides a separation between
       first and second sentences and also their sizes.
    """

    def __init__(self, sentences1, sentences2, sizes1, sizes2, labels):
        """
        :param sentences1: A 2D numpy array with sentences (the first in each pair)
            composed of token indices
        :param sentences2: Same as above for the second sentence in each pair
        :param sizes1: A 1D numpy array with the size of each sentence in the first group.
            Sentences should be filled with the PADDING token after that point
        :param sizes2: Same as above
        :param labels: 1D numpy array with labels as integers
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.sizes1 = sizes1
        self.sizes2 = sizes2
        self.labels = labels
        self.num_items = len(sentences1)

    def shuffle_data(self):
        """
        Shuffle all data using the same random sequence.
        :return:
        """
        shuffle_arrays([self.sentences1, self.sentences2,
                        self.sizes1, self.sizes2, self.labels])


def tokenize(text, lowercase=True):
    """
    Tokenize a piece of text using the Treebank tokenizer

    :param lowercase: also convert the text to lowercase
    :return: a list of strings
    """
    if lowercase:
        text = text.lower()
    return tokenizer.tokenize(text)


def tokenize_corpus(pairs):
    """
    Tokenize all pairs.

    :param pairs: a list of tuples (sent1, sent2, relation)
    :return: a list of tuples as in pairs, except both sentences are now lists
        of tokens
    """
    tokenized_pairs = []
    for sent1, sent2, label in pairs:
        tokens1 = tokenize(sent1)
        tokens2 = tokenize(sent2)
        tokenized_pairs.append((tokens1, tokens2, label))

    return tokenized_pairs


def count_corpus_tokens(pairs):
    """
    Examine all pairs ans extracts all tokens from both text and hypothesis.

    :param pairs: a list of tuples (sent1, sent2, relation) with tokenized
        sentences
    :return: a Counter of lowercase tokens
    """
    c = Counter()
    for sent1, sent2, _ in pairs:
        c.update(t.lower() for t in sent1)
        c.update(t.lower() for t in sent2)

    return c


def config_logger(verbose):
    """
    Setup basic logger configuration

    :param verbose: boolean
    :return:
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='%(message)s', level=level)


def get_logger(name):
    """
    Setup and return a simple logger.
    :return:
    """
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def shuffle_arrays(*arrays):
    """
    Shuffle all given arrays with the same RNG state.

    All shuffling is in-place, i.e., this function returns None.
    """
    rng_state = np.random.get_state()
    for array in arrays:
        np.random.shuffle(array)
        np.random.set_state(rng_state)


def convert_labels(pairs):
    """
    Return a numpy array representing the labels in `pairs`

    :param pairs: a list of tuples (_, _, label), with label as a string
    :return: a numpy array
    """
    return np.array([label_map[pair[2]] for pair in pairs], dtype=np.int32)


def find_max_len(pairs, index):
    """
    Find the maximum length among tokenized sentences

    :param pairs: list of tuples (sent1, sent2, label) with tokenized
        sentences
    :param index: whether to check the first sentence in each tuple
        or the second (0 or 1)
    :return: an integer
    """
    return max(len(pair[index]) for pair in pairs)


def generate_dataset(pairs, word_dict, max_len1=None, max_len2=None):
    """
    Generate and return a RTEDataset object for storing the data in numpy format.
    :param pairs: list of tokenized tuples (sent1, sent2, label)
    :param word_dict: a dictionary mapping words to indices
    :param max_len1: the maximum length that arrays for sentence 1
        should have (i.e., time steps for an LSTM). If None, it
        is computed from the data.
    :param max_len2: same as max_len1 for sentence 2
    :return: RTEDataset
    """
    tokens1 = [pair[0] for pair in pairs]
    tokens2 = [pair[1] for pair in pairs]
    sentences1, sizes1 = _convert_pairs_to_indices(tokens1, word_dict,
                                                   max_len1)
    sentences2, sizes2 = _convert_pairs_to_indices(tokens2, word_dict,
                                                   max_len2)
    labels = convert_labels(pairs)

    return RTEDataset(sentences1, sentences2, sizes1, sizes2, labels)


def _convert_pairs_to_indices(sentences, word_dict, max_len=None,
                              use_null=True):
    """
    Convert all pairs to their indices in the vector space.

    The maximum length of the arrays will be 1 more than the actual
    maximum of tokens when using the NULL symbol.

    :return: a tuple with a 2-d numpy array for the sentences and
        a 1-d array with their sizes
    """
    sizes = np.array([len(sent) for sent in sentences])
    if use_null:
        sizes += 1
        if max_len is not None:
            max_len += 1

    if max_len is None:
        max_len = sizes.max()

    shape = (len(sentences), max_len)
    array = np.full(shape, word_dict[PADDING], dtype=np.int32)

    for i, sent in enumerate(sentences):
        indices = [word_dict[token] for token in sent]

        if use_null:
            indices = [word_dict[GO]] + indices

        array[i, :len(indices)] = indices

    return (array, sizes)


def load_parameters(dirname):
    """
    Load a dictionary containing the parameters used to train an instance
    of the autoencoder.

    :param dirname: the path to the directory with the model files.
    :return: a Python dictionary
    """
    filename = os.path.join(dirname, 'model-params.json')
    with open(filename, 'rb') as f:
        data = json.load(f)

    return data


def get_sentence_sizes(pairs):
    """
    Count the sizes of all sentences in the pairs
    :param pairs: a list of tuples (sent1, sent2, _). They must be
        tokenized
    :return: a tuple (sizes1, sizes2), as two numpy arrays
    """
    sizes1 = np.array([len(pair[0]) for pair in pairs])
    sizes2 = np.array([len(pair[1]) for pair in pairs])
    return (sizes1, sizes2)


def get_max_sentence_sizes(pairs1, pairs2):
    """
    Find the maximum length among the first and second sentences in both
    pairs1 and pairs2. The two lists of pairs could be the train and validation
    sets

    :return: a tuple (max_len_sentence1, max_len_sentence2)
    """
    train_sizes1, train_sizes2 = get_sentence_sizes(pairs1)
    valid_sizes1, valid_sizes2 = get_sentence_sizes(pairs2)
    train_max1 = max(train_sizes1)
    valid_max1 = max(valid_sizes1)
    max_size1 = max(train_max1, valid_max1)
    train_max2 = max(train_sizes2)
    valid_max2 = max(valid_sizes2)
    max_size2 = max(train_max2, valid_max2)

    return max_size1, max_size2
