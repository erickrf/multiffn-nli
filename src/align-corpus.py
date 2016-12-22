# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import argparse
import nltk
from nltk.corpus import wordnet as wn
import json

"""
Script to read the SNLI corpus and produce a file with naive
alignments between words.
"""


wnl = nltk.stem.WordNetLemmatizer()


def map_wn_pos(pos):
    """
    Map a Penn Treebank POS tag to a WordNet style one
    """
    if pos in ['NN', 'NNS']:  # not NNP!
        return 'n'
    elif pos.startswith('JJ'):
        return 'a'
    elif pos.startswith('RB'):
        return 'r'
    elif pos.startswith('VB'):
        return 'v'
    else:
        return None


def read_words_pos(parse):
    """
    Read the given parse string and return a list of tuples
    (token, pos_tag) in lower case.
    """
    parse = parse.lower()
    tree = nltk.Tree.fromstring(parse)
    word_pos = [(word, map_wn_pos(pos)) for word, pos in tree.pos()]
    return word_pos


def get_lemma(word, pos):
    """
    Return the lemma of given word and pos
    """
    if pos is None:
        return word
    return wnl.lemmatize(word, pos)


def same_synset(word1, pos1, word2, pos2):
    """Check if two words share at least one synset in wordnet"""
    if pos1 is None or pos2 is None:
        return False

    synset_set1 = set(wn.synsets(word1, pos1))
    for synset2 in wn.synsets(word2, pos2):
        if synset2 in synset_set1:
            return True

    return False


def align(words_pos1, words_pos2):
    """
    Return the token-level alignments between two sentences
    :param words_pos1: list of tuples (token, pos)
    :param words_pos2: list of tuples (token, pos)
    :return:
    """
    alignments = []
    lemmas1 = [get_lemma(word, pos) for word, pos in words_pos1]
    lemmas2 = [get_lemma(word, pos) for word, pos in words_pos2]

    for i, lemma1 in enumerate(lemmas1):
        pos1 = words_pos1[i][1]
        for j, lemma2 in enumerate(lemmas2):
            pos2 = words_pos2[j][1]
            if lemma1 == lemma2 or same_synset(lemma1, pos1,
                                               lemma2, pos2):
                # todo: check hypernyms
                alignments.append((i, j))

    return alignments


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Path to corpus in JSONL format')
    parser.add_argument('output', help='Path to write output (JSONL)')
    args = parser.parse_args()

    i = 0
    with open(args.input, 'rb') as fin, open(args.output, 'wb') as fout:
        for line in fin:
            line = line.decode('utf-8')
            data = json.loads(line)

            tree1 = nltk.Tree.fromstring(data['sentence1_parse'])
            tree2 = nltk.Tree.fromstring(data['sentence2_parse'])

            words_pos1 = [(word.lower(), map_wn_pos(pos))
                          for word, pos in tree1.pos()]
            words_pos2 = [(word.lower(), map_wn_pos(pos))
                          for word, pos in tree2.pos()]

            alignments = align(words_pos1, words_pos2)
            i += 1
            if i % 1000 == 0:
                print('Read %d lines' % i)

            output_data = {'sentence1': tree1.leaves(),
                           'sentence2': tree2.leaves(),
                           'alignment': alignments}
            fout.write(json.dumps(output_data).encode('utf-8'))
            fout.write('\n')
