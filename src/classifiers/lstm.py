# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf

from multimlp import MultiFeedForwardClassifier


class LSTMClassifier(MultiFeedForwardClassifier):
    """
    Improvements over the multi feed forward classifier. This is mostly
    based on "Enhancing and Combining Sequential and Tree LSTM for
    Natural Language Inference", by Chen et al. (2016), using LSTMs
    instead of MLP networks.
    """
    def __init__(self, weights, bias, *args, **kwars):
        if weights is not None and bias is not None:
            self._initialize_rnn(weights, bias)
            self.pre_initialized = True
        else:
            self.pre_initialized = False

        super(LSTMClassifier, self).__init__(*args, **kwars)

    def _extra_init(self):
        """
        Extra initialization stuff inside the constructor
        """
        #TODO: use bidirectional LSTMs, as in the original implementation
        if self.use_intra:
            msg = 'LSTMClassifier is not compatible with intra attention'
            raise ValueError(msg)

        initializer = tf.contrib.layers.xavier_initializer()
        self.lstm = tf.nn.rnn_cell.LSTMCell(self.num_units,
                                            initializer=initializer)

    def _initialize_rnn(self, weights, bias):
        """
        Initialize the weights and bias of the projection RNN (used in the
        attend phase) with preset values.

        :param weights: 2d numpy array
        :param bias: 1d numpy array
        :param num_units: python int
        """
        # this implementation is hacky and I'd like to replace it with
        # something better. but right now I have no idea how.
        with tf.variable_scope('inter-attention'):
            _ = tf.get_variable('LSTMCell/W_0', initializer=tf.constant(weights))
            _ = tf.get_variable('LSTMCell/B', initializer=tf.constant(bias))

    def _num_units_on_aggregate(self):
        """
        Return the number of units used by the network when computing
        the aggregated representation of the two sentences.
        """
        return 2 * self.lstm.output_size

    def _transformation_attend(self, sentence, num_units, length,
                               reuse_weights=False):
        """
        Transform sentences using the RNN.
        """
        expected_num = self.num_units if self.project_input \
            else self.embedding_size

        assert num_units == expected_num, \
            'Expected sentences with dimension %d, got %d instead:' % \
            (expected_num, num_units)

        if self.pre_initialized:
            reuse_weights = True
        return self._apply_lstm(sentence, length, self.attend_scope,
                                reuse_weights)

    def _transformation_compare(self, sentence, num_units, length,
                                reuse_weights=False):
        """
        Perform the sentence comparison using the RNN.
        """
        # return super(LSTMClassifier, self)._transformation_compare(sentence, num_units,
        #                                                            length, reuse_weights)
        return self._apply_lstm(sentence, length, self.compare_scope,
                                reuse_weights)

    def _apply_lstm(self, inputs, length, scope=None, reuse_weights=False):
        """
        Apply the given RNN cell to the given sentences, taking care of
        weight reusing.
        """
        scope_name = scope or 'lstm'
        with tf.variable_scope(scope_name, reuse=reuse_weights) as lstm_scope:
            outputs, _ = tf.nn.dynamic_rnn(self.lstm, inputs,
                                           dtype=tf.float32,
                                           sequence_length=length,
                                           scope=lstm_scope)
        return outputs
