# -*- coding: utf-8 -*-

import tensorflow as tf

from multimlp import MultiFeedForwardClassifier
from trainable import Trainable, get_weights_and_biases


def clip_alignments(alignments, size1, size2):
    """
    Clip an alignment matrix to the length of clipped sentences
    (the maximum size in the batch).

    :param alignments: alignment tensor (batch, sent1_size, sent2_size)
    :param size1: tensor with shape (batch)
    :param size2: tensor with shape (batch)
    :return: tensor with shape (batch, max(size1), max(size2))
    """
    max_size1 = tf.reduce_max(size1)
    max_size2 = tf.reduce_max(size2)
    clipped_alignments = tf.slice(alignments, [0, 0, 0],
                                  tf.pack([-1, max_size1, max_size2]))
    return clipped_alignments


class AlignPretrainer(Trainable):
    """
    Class to wrap an instance of a decomposable neural classifier
    and pretrain its weights for the alignment subtask.
    """

    scope_name = 'aligner'

    def __init__(self, classifier):
        """
        :param classifier: an instance of MultiFeedForwardClassifier
        :type classifier: MultiFeedForwardClassifier
        """
        self.classifier = classifier
        assert isinstance(self.classifier, MultiFeedForwardClassifier)
        # alignments has shape (batch, length1, length2)
        self.alignments = tf.placeholder(tf.float32, [None, None, None],
                                         name='gold_alignments')
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        self.l2_constant = tf.placeholder(tf.float32, [], 'l2_constant')
        self.clip_value = tf.placeholder(tf.float32, [], 'clip_norm')

        # the weights are trained to match the attention in the gold label
        # one word in sentence one can be aligned to one or more in sentence
        # two. Cases where there is no alignment are encoded as alignment
        # to a NULL symbol
        logits = self.classifier.raw_attentions

        with tf.variable_scope(self.scope_name):
            # ensure the sum of each row yields 1
            alignments = clip_alignments(self.alignments,
                                         classifier.sentence1_size,
                                         classifier.sentence2_size)
            alignments = alignments / tf.reduce_sum(alignments, 2, keep_dims=True)
            cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                                alignments)
            alignment_loss = tf.reduce_mean(cross_ent)

            weights = [v for v in tf.global_variables() if 'weight' in v.name]
            l2_partial_sum = sum([tf.nn.l2_loss(weight) for weight in weights])
            l2_loss = tf.mul(self.l2_constant, l2_partial_sum, 'l2_loss')
            self.loss = tf.add(alignment_loss, l2_loss, 'loss')
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
            self.train_op = optimizer.apply_gradients(zip(gradients, v))

    def save(self, save_dir, session, saver):
        """
        Save the contained classifier.
        """
        self.classifier.save(save_dir, session, saver)

    def initialize(self, session):
        """
        Initialize tensorflow variables used by the aligner's graph.
        """
        vars_ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_name)
        session.run(tf.variables_initializer(vars_))

    def _create_batch_feed(self, sentence1, sentence2, size1, size2, label,
                           learning_rate, dropout_keep, l2, clip_value):

        feeds = {self.classifier.sentence1: sentence1,
                 self.classifier.sentence2: sentence2,
                 self.classifier.sentence1_size: size1,
                 self.classifier.sentence2_size: size2,
                 self.alignments: label,
                 self.learning_rate: learning_rate,
                 self.classifier.dropout_keep: dropout_keep,
                 self.l2_constant: l2,
                 self.clip_value: clip_value
                 }

        return feeds

    def _run_on_validation(self, session, feeds):
        loss = session.run(self.loss, feeds)
        msg = 'Validation loss: %f' % loss
        return loss, msg

    def train(self, session, train_dataset, valid_dataset, save_dir,
              learning_rate, num_epochs, batch_size, dropout_keep, l2,
              clip_norm, report_interval=100):
        super(AlignPretrainer, self)._train(session, get_weights_and_biases(),
                                            save_dir, train_dataset, valid_dataset,
                                            learning_rate, num_epochs, batch_size,
                                            dropout_keep, l2, clip_norm, report_interval)

