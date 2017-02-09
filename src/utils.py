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
import tensorflow as tf
from collections import Counter
from nltk.tokenize.regexp import RegexpTokenizer

import classifiers

tokenizer = nltk.tokenize.TreebankWordTokenizer()
UNKNOWN = '**UNK**'
PADDING = '**PAD**'
GO = '**GO**'  # it's called "GO" but actually serves as a null alignment


class RTEDataset(object):
    """
    Class for better organizing a data set. It provides a separation between
       first and second sentences and also their sizes.
    """

    def __init__(self, sentences1, sentences2, sizes1, sizes2, labels):
        """
        :param sentences1: A 2D numpy array with sentences (the first in each
            pair) composed of token indices
        :param sentences2: Same as above for the second sentence in each pair
        :param sizes1: A 1D numpy array with the size of each sentence in the
            first group. Sentences should be filled with the PADDING token after
            that point
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

    def get_batch(self, from_, to):
        """
        Return an RTEDataset object with the subset of the data contained in
        the given interval. Note that the actual number of items may be less
        than (`to` - `from_`) if there are not enough of them.

        :param from_: which position to start from
        :param to: which position to end
        :return: an RTEDataset object
        """
        if from_ == 0 and to >= self.num_items:
            return self

        subset = RTEDataset(self.sentences1[from_:to],
                            self.sentences2[from_:to],
                            self.sizes1[from_:to],
                            self.sizes2[from_:to],
                            self.labels[from_:to])
        return subset


def get_tokenizer(language):
    """
    Return the tokenizer function according to the language.
    """
    language = language.lower()
    if language == 'en':
        tokenize = tokenize_english
    elif language == 'pt':
        tokenize = tokenize_portuguese
    else:
        ValueError('Unsupported language: %s' % language)

    return tokenize


def tokenize_english(text):
    """
    Tokenize a piece of text using the Treebank tokenizer

    :return: a list of strings
    """
    return tokenizer.tokenize(text)


def tokenize_portuguese(text):
    """
    Tokenize the given sentence in Portuguese. The tokenization is done in
    conformity  with Universal Treebanks (at least it attempts so).

    :param text: text to be tokenized, as a string
    """
    tokenizer_regexp = ur'''(?ux)
    # the order of the patterns is important!!
    # more structured patterns come first
    [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|    # emails
    (?:[\#@]\w+)|                     # Hashtags and twitter user names
    (?:[^\W\d_]\.)+|                  # one letter abbreviations, e.g. E.U.A.
    (?:[DSds][Rr][Aa]?)\.|            # common abbreviations such as dr., sr., sra., dra.
    \b\d+(?:[-:.,]\w+)*(?:[.,]\d+)?\b|
        # numbers in format 999.999.999,999, or hyphens to alphanumerics
    \.{3,}|                           # ellipsis or sequences of dots
    (?:\w+(?:\.\w+|-\d+)*)|           # words with dots and numbers, possibly followed by hyphen number
    -+|                               # any sequence of dashes
    \S                                # any non-space character
    '''
    tokenizer = RegexpTokenizer(tokenizer_regexp)

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
        tokens1 = tokenize_english(sent1)
        tokens2 = tokenize_english(sent2)
        tokenized_pairs.append((tokens1, tokens2, label))

    return tokenized_pairs


def count_parameters():
    """
    Count the number of trainable tensorflow parameters loaded in
    the current graph.
    """
    total_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_params = 1
        for dim in shape:
            variable_params *= dim.value
        logging.debug('%s: %d params' % (variable.name, variable_params))
        total_params += variable_params
    return total_params


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


def get_logger(name='logger'):
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


def get_model_class(params):
    """
    Return the class of the model object

    :param params: saved parameter dictionary
    :return: a subclass of classifiers.DecomposableNLIModel
    """
    if params.get('model') == 'lstm':
        model_class = classifiers.LSTMClassifier
    else:
        model_class = classifiers.MultiFeedForwardClassifier

    assert issubclass(model_class, classifiers.DecomposableNLIModel)
    return model_class


def create_label_dict(pairs):
    """
    Return a dictionary mapping the labels found in `pairs` to numbers
    :param pairs: a list of tuples (_, _, label), with label as a string
    :return: a dict
    """
    labels = set(pair[2] for pair in pairs)
    mapping = zip(labels, range(len(labels)))
    return dict(mapping)


def convert_labels(pairs, label_map):
    """
    Return a numpy array representing the labels in `pairs`

    :param pairs: a list of tuples (_, _, label), with label as a string
    :param label_map: dictionary mapping label strings to numbers
    :return: a numpy array
    """
    return np.array([label_map[pair[2]] for pair in pairs], dtype=np.int32)


def create_dataset(pairs, word_dict, label_dict=None,
                   max_len1=None, max_len2=None):
    """
    Generate and return a RTEDataset object for storing the data in numpy format.

    :param pairs: list of tokenized tuples (sent1, sent2, label)
    :param word_dict: a dictionary mapping words to indices
    :param label_dict: a dictionary mapping labels to numbers. If None,
        labels are ignored.
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
    if label_dict is not None:
        labels = convert_labels(pairs, label_dict)
    else:
        labels = None

    return RTEDataset(sentences1, sentences2, sizes1, sizes2, labels)


def _convert_pairs_to_indices(sentences, word_dict, max_len=None,
                              use_null=True):
    """
    Convert all pairs to their indices in the vector space.

    The maximum length of the arrays will be 1 more than the actual
    maximum of tokens when using the NULL symbol.

    :param sentences: list of lists of tokens
    :param word_dict: mapping of tokens to indices in the embeddings
    :param max_len: maximum allowed sentence length. If None, the
        longest sentence will be the maximum
    :param use_null: prepend a null symbol at the beginning of each
        sentence
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

    return array, sizes


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


def normalize_embeddings(embeddings):
    """
    Normalize the embeddings to have norm 1.
    :param embeddings: 2-d numpy array
    :return: normalized embeddings
    """
    # normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1).reshape((-1, 1))
    return embeddings / norms
