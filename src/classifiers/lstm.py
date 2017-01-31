# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf

from decomposable import DecomposableNLIModel


class LSTMClassifier(DecomposableNLIModel):
    """
    Improvements over the multi feed forward classifier. This is mostly
    based on "Enhancing and Combining Sequential and Tree LSTM for
    Natural Language Inference", by Chen et al. (2016), using LSTMs
    instead of MLP networks.
    """
    def __init__(self, *args, **kwars):
        super(LSTMClassifier, self).__init__(*args, **kwars)

    def _extra_init(self):
        """
        Extra initialization stuff inside the constructor
        """
        initializer = tf.contrib.layers.xavier_initializer()
        self.lstm = tf.nn.rnn_cell.LSTMCell(self.num_units,
                                            initializer=initializer)

    def _num_inputs_on_aggregate(self):
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

        return self._apply_lstm(sentence, length, self.attend_scope,
                                reuse_weights)

    def _transformation_compare(self, sentence, num_units, length,
                                reuse_weights=False):
        """
        Perform the sentence comparison using the RNN.
        """
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
