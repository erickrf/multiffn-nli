# -*- coding: utf-8 -*-

"""
Functions for reading data in a format suitable for the networks.
"""

import json
import os
import numpy as np
import nltk
from collections import defaultdict

import utils


def write_word_dict(word_dict, dirname):
    """
    Write the word dictionary to a file.

    It is understood that unknown words are mapped to 0.
    """
    words = [word for word in word_dict.keys() if word_dict[word] != 0]
    sorted_words = sorted(words, key=lambda x: word_dict[x])
    text = '\n'.join(sorted_words)
    path = os.path.join(dirname, 'word-dict.txt')
    with open(path, 'wb') as f:
        f.write(text.encode('utf-8'))


def read_word_dict(dirname):
    """
    Read a file with a list of words and generate a defaultdict from it.
    """
    filename = os.path.join(dirname, 'word-dict.txt')
    with open(filename, 'rb') as f:
        text = f.read().decode('utf-8')

    words = text.splitlines()
    index_range = range(1, len(words) + 1)
    return defaultdict(int, zip(words, index_range))


def load_text_embeddings(path, generate_padding=True, generate_go=True):
    """
    Load any embedding model written as text, in the format:
    word[space or tab][values separated by space or tab]

    :param path: path to embeddings file
    :param generate_padding: generate a zero-based padding vector
    :param generate_go: generate a random vector for the GO symbol
    :return: a tuple (defaultdict, array) The dict maps words to indices.
        Unknown words are mapped to 0.
    """
    words = defaultdict(int)

    # start from index 1 and reserve 0 for unknown
    index = 1
    vectors = []
    with open(path, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            line = line.strip()
            if line == '':
                continue

            fields = line.split()
            word = fields[0]
            words[word] = index
            index += 1
            vector = np.array([float(x) for x in fields[1:]], dtype=np.float32)
            vectors.append(vector)

    vector_size = len(vectors[0])
    vectors.insert(0, np.random.uniform(-0.1, 0.1, vector_size))
    index = len(vectors)

    if generate_padding:
        words[utils.PADDING] = index
        vectors.append(np.random.uniform(-0.1, 0.1, vector_size))
        # vectors.append(np.zeros(vector_size))
        index += 1
    if generate_go:
        words[utils.GO] = index
        vectors.append(np.random.uniform(-0.1, 0.1, vector_size))
        index += 1

    embeddings = np.array(vectors, dtype=np.float32)

    return words, embeddings


def read_snli(filename):
    """
    Read a JSONL or TSV file with the SNLI corpus

    :param filename: path to the file
    :return: a list of tuples (first_sent, second_sent, label)
    """
    # we are only interested in the actual sentences + gold label
    # the corpus files has a few more things
    useful_data = []

    # the SNLI corpus has one JSON object per line
    with open(filename, 'rb') as f:

        if filename.endswith('.tsv') or filename.endswith('.txt'):
            f.seek(0)
            for line in f:
                line = line.decode('utf-8').strip()
                sent1, sent2, label = line.split('\t')
                if label == '-':
                    continue
                tokens1 = sent1.split()
                tokens2 = sent2.split()
                useful_data.append((tokens1, tokens2, label))
        else:
            for line in f:
                line = unicode(line, 'utf-8')
                data = json.loads(line)

                if data['gold_label'] == '-':
                    # ignore items without a gold label
                    continue

                sentence1_parse = data['sentence1_parse']
                sentence2_parse = data['sentence2_parse']
                label = data['gold_label']

                tree1 = nltk.Tree.fromstring(sentence1_parse.lower())
                tree2 = nltk.Tree.fromstring(sentence2_parse.lower())
                tokens1 = tree1.leaves()
                tokens2 = tree2.leaves()
                t = (tokens1, tokens2, label)
                useful_data.append(t)

    return useful_data

