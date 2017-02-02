# -*- coding: utf-8 -*-

from __future__ import print_function, division

import logging

import numpy as np
import tensorflow as tf

import utils
from decomposable import DecomposableNLIModel, attention_softmax3d,\
    get_weights_and_biases


class MultiFeedForwardClassifier(DecomposableNLIModel):
    """
    Implementation of the multi feed forward network model described in
    the paper "A Decomposable Attention Model for Natural Language
    Inference" by Parikh et al., 2016.

    It applies feedforward MLPs to combinations of parts of the two sentences,
    without any recurrent structure.
    """
    def __init__(self, num_units, num_classes, vocab_size, embedding_size,
                 training=True, project_input=True, optimizer='adagrad',
                 use_intra_attention=False, distance_biases=10):
        """
        Create the model based on MLP networks.

        :param num_units: size of the networks
        :param num_classes: number of classes in the problem
        :param vocab_size: size of the vocabulary
        :param embedding_size: size of each word embedding
        :param use_intra_attention: whether to use intra-attention model
        :param training: whether to create training tensors (optimizer)
        :param project_input: whether to project input embeddings to a
            different dimensionality
        :param distance_biases: number of different distances with biases used
            in the intra-attention model
        """
        self.use_intra = use_intra_attention
        self.distance_biases = distance_biases

        super(MultiFeedForwardClassifier, self).\
            __init__(num_units, num_classes, vocab_size, embedding_size,
                     training, project_input, optimizer)

    def _transformation_input(self, inputs, reuse_weights=False):
        """
        Apply any transformations to the input embeddings

        :param inputs: a tensor with shape (batch, time_steps, embeddings)
        :return: a tensor of the same shape of the input
        """
        transformed = super(MultiFeedForwardClassifier, self).\
            _transformation_input(inputs, reuse_weights)

        if self.use_intra:
            # here, repr's have shape (batch , time_steps, 2*num_units)
            transformed = self.compute_intra_attention(transformed,
                                                       reuse_weights)
            self.representation_size *= 2

        return transformed

    def _get_params_to_save(self):
        params = super(MultiFeedForwardClassifier, self)._get_params_to_save()
        params['use_intra'] = self.use_intra
        params['distance_biases'] = self.distance_biases
        return params

    @classmethod
    def _init_from_load(cls, params, training):
        return cls(params['num_units'], params['num_classes'],
                   params['vocab_size'], params['embedding_size'],
                   project_input=params['project_input'], training=training,
                   use_intra_attention=params['use_intra'],
                   distance_biases=params['distance_biases'])

    def _get_distance_biases(self, time_steps, reuse_weights=False):
        """
        Return a 2-d tensor with the values of the distance biases to be applied
        on the intra-attention matrix of size sentence_size
        :param time_steps: tensor scalar
        :return: 2-d tensor (time_steps, time_steps)
        """
        with tf.variable_scope('distance-bias', reuse=reuse_weights):
            # this is d_{i-j}
            distance_bias = tf.get_variable('dist_bias', [self.distance_biases],
                                            initializer=tf.zeros_initializer)

            # messy tensor manipulation for indexing the biases
            r = tf.range(0, time_steps)
            r_matrix = tf.tile(tf.reshape(r, [1, -1]), tf.pack([time_steps, 1]))
            raw_inds = r_matrix - tf.reshape(r, [-1, 1])
            clipped_inds = tf.clip_by_value(raw_inds, 0,
                                            self.distance_biases - 1)
            values = tf.nn.embedding_lookup(distance_bias, clipped_inds)

        return values

    def compute_intra_attention(self, sentence, reuse_weights=False):
        """
        Compute the intra attention of a sentence. It returns a concatenation
        of the original sentence with its attended output.

        :param sentence: tensor in shape (batch, time_steps, num_units)
        :return: a tensor in shape (batch, time_steps, 2*num_units)
        """
        time_steps = tf.shape(sentence)[1]
        with tf.variable_scope('intra-attention') as scope:
            # this is F_intra in the paper
            # f_intra1 is (batch, time_steps, num_units) and
            # f_intra1_t is (batch, num_units, time_steps)
            f_intra = self._apply_feedforward(sentence, self.num_units,
                                              scope,
                                              reuse_weights=reuse_weights)
            f_intra_t = tf.transpose(f_intra, [0, 2, 1])

            # these are f_ij
            # raw_attentions is (batch, time_steps, time_steps)
            raw_attentions = tf.batch_matmul(f_intra, f_intra_t)

            # bias has shape (time_steps, time_steps)
            bias = self._get_distance_biases(time_steps,
                                             reuse_weights=reuse_weights)

            # bias is broadcast along batches
            raw_attentions += bias
            attentions = attention_softmax3d(raw_attentions)
            attended = tf.batch_matmul(attentions, sentence)

        return tf.concat(2, [sentence, attended])

    def _transformation_attend(self, sentence, num_units, length,
                               reuse_weights=False):
        """
        Apply the transformation on each sentence before attending over each
        other. In the original model, it is a two layer feed forward network.

        :param sentence: a tensor with shape (batch, time_steps, num_units)
        :param num_units: a python int indicating the third dimension of
            sentence
        :param length: real length of the sentence. Not used in this class.
        :param reuse_weights: whether to reuse weights inside this scope
        :return: a tensor with shape (batch, time_steps, num_units)
        """
        return self._apply_feedforward(sentence, num_units, self.attend_scope,
                                       reuse_weights)

    def _transformation_compare(self, sentence, num_units, length,
                                reuse_weights=False):
        """
        Apply the transformation on each attended token before comparing.
        In the original model, it is a two layer feed forward network.

        :param sentence: a tensor with shape (batch, time_steps, num_units)
        :param num_units: a python int indicating the third dimension of
            sentence
        :param length: real length of the sentence. Not used in this class.
        :param reuse_weights: whether to reuse weights inside this scope
        :return: a tensor with shape (batch, time_steps, num_units)
        """
        return self._apply_feedforward(sentence, num_units, self.compare_scope,
                                       reuse_weights)

    def evaluate(self, session, dataset, return_answers, batch_size=5000):
        """
        Run the model on the given dataset

        :param session: tensorflow session
        :param dataset: the dataset object
        :type dataset: utils.RTEDataset
        :param return_answers: if True, also return the answers
        :param batch_size: how many items to run at a time. Relevant if you're
            evaluating the train set performance
        :return: if not return_answers, a tuple (loss, accuracy)
            or else (loss, accuracy, answers)
        """
        if return_answers:
            ops = [self.loss, self.accuracy, self.answer]
        else:
            ops = [self.loss, self.accuracy]

        i = 0
        j = batch_size
        # store accuracies and losses weighted by the number of items in the
        # batch to take the correct average in the end
        weighted_accuracies = []
        weighted_losses = []
        answers = []
        while i <= dataset.num_items:
            subset = dataset.get_batch(i, j)

            last = i + subset.num_items
            logging.debug('Evaluating items between %d and %d' % (i, last))
            i = j
            j += batch_size

            feeds = self._create_batch_feed(subset, 0, 1, 0, 0)
            results = session.run(ops, feeds)
            weighted_losses.append(results[0] * subset.num_items)
            weighted_accuracies.append(results[1] * subset.num_items)
            if return_answers:
                answers.append(results[2])

        avg_accuracy = np.sum(weighted_accuracies) / dataset.num_items
        avg_loss = np.sum(weighted_losses) / dataset.num_items
        ret = [avg_loss, avg_accuracy]
        if return_answers:
            all_answers = np.concatenate(answers)
            ret.append(all_answers)

        return ret
