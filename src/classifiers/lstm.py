# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf

from multimlp import MultiFeedForwardClassifier


def _initialize_rnn(weights, bias):
    """
    Initialize the weights and bias of the projection RNN (used in the
    attend phase) with preset values.

    :param weights: 2d numpy array
    :param bias: 1d numpy array
    """
    # this implementation is hacky and I'd like to replace it with
    # something better. but right now I have no idea how.
    with tf.variable_scope('inter-attention'):
        _ = tf.get_variable('LSTMCell/W_0', initializer=tf.constant(weights))
        _ = tf.get_variable('LSTMCell/B', initializer=tf.constant(bias))


class LSTMClassifier(MultiFeedForwardClassifier):
    """
    Improvements over the multi feed forward classifier. This is mostly
    based on "Enhancing and Combining Sequential and Tree LSTM for
    Natural Language Inference", by Chen et al. (2016), using LSTMs
    instead of MLP networks.
    """
    def __init__(self, weights, bias, *args, **kwars):
        """
        Initialize the LSTM, possibly with pretrained weights
        :param weights: LSTM weights to initialize the inter-attention
            module. It can be None.
        :param bias: LSTM biases to initialize the inter-attention
            module. It can be None
        :param args:
        :param kwars:
        """
        if weights is not None and bias is not None:
            _initialize_rnn(weights, bias)
            self.pre_initialized = True
        else:
            self.pre_initialized = False

        super(LSTMClassifier, self).__init__(*args, **kwars)

    def _extra_init(self):
        """
        Extra initialization stuff inside the constructor
        """
        if self.use_intra:
            msg = 'LSTMClassifier is not compatible with intra attention'
            raise ValueError(msg)

        initializer = tf.contrib.layers.xavier_initializer()
        self.lstm = tf.nn.rnn_cell.LSTMCell(self.num_units,
                                            initializer=initializer)

    def _num_units_on_aggregate(self):
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

        if self.pre_initialized:
            reuse_weights = True
        return self._apply_lstm(after_dropout, length, self.attend_scope,
                                reuse_weights)

    @classmethod
    def _init_from_load(cls, params, training):
        return cls(None, None, params['num_units'],
                   params['num_classes'], params['vocab_size'],
                   params['embedding_size'], training=training,
                   project_input=params['project_input'])

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
            # outputs, _ = tf.nn.dynamic_rnn(self.lstm, inputs,
            #                                dtype=tf.float32,
            #                                sequence_length=length,
            #                                scope=lstm_scope)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.lstm, self.lstm,
                                                         inputs,
                                                         dtype=tf.float32,
                                                         sequence_length=length,
                                                         scope=lstm_scope)
            output_fw, output_bw = outputs
            concat_outputs = tf.concat(2, [output_fw, output_bw])
        return concat_outputs
