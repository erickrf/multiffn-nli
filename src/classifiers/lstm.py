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
    def _extra_init(self):
        """
        Extra initialization stuff inside the constructor
        """
        #TODO: use bidirectional LSTMs, as in the original implementation
        if self.use_intra:
            msg = 'LSTMClassifier is not compatible with intra attention'
            raise ValueError(msg)

        initializer = tf.contrib.layers.xavier_initializer()
        self.attention_lstm = tf.nn.rnn_cell.LSTMCell(2*self.num_units,
                                                      initializer=initializer)
        self.comparison_lstm = tf.nn.rnn_cell.LSTMCell(2*self.num_units,
                                                       initializer=initializer)

    def _num_units_on_aggregate(self):
        """
        Return the number of units used by the network when computing
        the aggregated representation of the two sentences.
        """
        return 2 * self.comparison_lstm.output_size

    def _transformation_attend(self, sentence, num_units, length,
                               reuse_weights=False):
        """
        Transform sentences using the RNN.
        """
        assert num_units == self.num_units, \
            'Expected sentences with dimension %d, got %d instead:' % \
            (self.num_units, num_units)

        return self._apply_lstm(sentence, length, self.attention_lstm,
                                reuse_weights)

    def _transformation_compare(self, sentence, num_units, length,
                                reuse_weights=False):
        """
        Perform the sentence comparison using the RNN.
        """
        assert num_units == 2 * self.num_units, \
            'Expected sentences with dimension %d, got %d instead:' % \
            (self.num_units, num_units)

        return self._apply_lstm(sentence, length, self.comparison_lstm,
                                reuse_weights)

    def _apply_lstm(self, inputs, length, cell, reuse_weights=False):
        """
        Apply the given RNN cell to the given sentences, taking care of
        weight reusing.
        """
        with tf.variable_scope('lstm', reuse=reuse_weights) as lstm_scope:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs,
                                           dtype=tf.float32,
                                           sequence_length=length,
                                           scope=lstm_scope)
        return outputs
