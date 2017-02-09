# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf

from decomposable import DecomposableNLIModel, mask_3d


class LSTMClassifier(DecomposableNLIModel):
    """
    Improvements over the multi feed forward classifier. This is mostly
    based on "Enhancing and Combining Sequential and Tree LSTM for
    Natural Language Inference", by Chen et al. (2016), using LSTMs
    instead of MLP networks.
    """
    def __init__(self, *args, **kwars):
        """
        Initialize the LSTM.

        :param args: args passed to DecomposableNLIModel
        :param kwars: kwargs passed to DecomposableNLIModel
        """
        super(LSTMClassifier, self).__init__(*args, **kwars)

    def _create_aggregate_input(self, v1, v2):
        """
        Create and return the input to the aggregate step.

        :param v1: tensor with shape (batch, time_steps, num_units)
        :param v2: tensor with shape (batch, time_steps, num_units)
        :return: a tensor with shape (batch, num_aggregate_inputs)
        """
        # sum over time steps; resulting shape is (batch, num_units)
        v1 = mask_3d(v1, self.sentence1_size, 0, 1)
        v2 = mask_3d(v2, self.sentence2_size, 0, 1)
        v1_sum = tf.reduce_sum(v1, [1])
        v2_sum = tf.reduce_sum(v2, [1])
        v1_max = tf.reduce_max(v1, [1])
        v2_max = tf.reduce_max(v2, [1])
        return tf.concat(1, [v1_sum, v2_sum, v1_max, v2_max])

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
        # 2 directions * 4 pools from the v1/v2 (v1/2 max, v1/2 mean)
        return 8 * self.lstm.output_size

    def _transformation_attend(self, sentence, num_units, length,
                               reuse_weights=False):
        """
        Transform sentences using the RNN.
        :param num_units: the number of units at each time step
        :param length: the total number of items in a sentence
        """
        expected_num = self.num_units if self.project_input \
            else self.embedding_size

        assert num_units == expected_num, \
            'Expected sentences with dimension %d, got %d instead:' % \
            (expected_num, num_units)

        after_dropout = tf.nn.dropout(sentence, self.dropout_keep)
        return self._apply_lstm(after_dropout, length, self.attend_scope,
                                reuse_weights)

    @classmethod
    def _init_from_load(cls, params, training):
        return cls(params['num_units'], params['num_classes'],
                   params['vocab_size'], params['embedding_size'],
                   training=training, project_input=params['project_input'])

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
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.lstm, self.lstm,
                                                         inputs,
                                                         dtype=tf.float32,
                                                         sequence_length=length,
                                                         scope=lstm_scope)
            output_fw, output_bw = outputs
            concat_outputs = tf.concat(2, [output_fw, output_bw])
        return concat_outputs
