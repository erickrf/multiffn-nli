# -*- coding: utf-8 -*-

import abc
import json
import logging
import os

import tensorflow as tf
import numpy as np

import utils


def attention_softmax3d(values):
    """
    Performs a softmax over the attention values.

    :param values: 3d tensor with raw values
    :return: 3d tensor, same shape as input
    """
    original_shape = tf.shape(values)
    num_units = original_shape[2]
    reshaped_values = tf.reshape(values, tf.stack([-1, num_units]))
    softmaxes = tf.nn.softmax(reshaped_values)
    return tf.reshape(softmaxes, original_shape)


def clip_sentence(sentence, sizes):
    """
    Clip the input sentence placeholders to the length of the longest one in the
    batch. This saves processing time.

    :param sentence: tensor with shape (batch, time_steps)
    :param sizes: tensor with shape (batch)
    :return: tensor with shape (batch, time_steps)
    """
    max_batch_size = tf.reduce_max(sizes)
    clipped_sent = tf.slice(sentence, [0, 0],
                            tf.stack([-1, max_batch_size]))
    return clipped_sent


def mask_3d(values, sentence_sizes, mask_value, dimension=2):
    """
    Given a batch of matrices, each with shape m x n, mask the values in each
    row after the positions indicated in sentence_sizes.

    This function is supposed to mask the last columns in the raw attention
    matrix (e_{i, j}) in cases where the sentence2 is smaller than the
    maximum.

    :param values: tensor with shape (batch_size, m, n)
    :param sentence_sizes: tensor with shape (batch_size) containing the
        sentence sizes that should be limited
    :param mask_value: scalar value to assign to items after sentence size
    :param dimension: over which dimension to mask values
    :return: a tensor with the same shape as `values`
    """
    if dimension == 1:
        values = tf.transpose(values, [0, 2, 1])
    time_steps1 = tf.shape(values)[1]
    time_steps2 = tf.shape(values)[2]

    ones = tf.ones_like(values, dtype=tf.float32)
    pad_values = mask_value * ones
    mask = tf.sequence_mask(sentence_sizes, time_steps2)

    # mask is (batch_size, sentence2_size). we have to tile it for 3d
    mask3d = tf.expand_dims(mask, 1)
    mask3d = tf.tile(mask3d, (1, time_steps1, 1))

    masked = tf.where(mask3d, values, pad_values)

    if dimension == 1:
        masked = tf.transpose(masked, [0, 2, 1])

    return masked


class DecomposableNLIModel(object):
    """
    Base abstract class for decomposable NLI models
    """
    abc.__metaclass__ = abc.ABCMeta

    def __init__(self, num_units, num_classes, vocab_size, embedding_size,
                 training=True, project_input=True, optimizer='adagrad'):
        """
        Create the model based on MLP networks.

        :param num_units: main dimension of the internal networks
        :param num_classes: number of possible classes
        :param vocab_size: size of the vocabulary
        :param embedding_size: size of each word embedding
        :param training: whether to create training tensors (optimizer)
        :param project_input: whether to project input embeddings to a
            different dimensionality
        """
        self.num_units = num_units
        self.num_classes = num_classes
        self.project_input = project_input

        # we have to supply the vocab size to allow validate_shape on the
        # embeddings variable, which is necessary down in the graph to determine
        # the shape of inputs at graph construction time
        self.embeddings_ph = tf.placeholder(tf.float32, (vocab_size,
                                                         embedding_size),
                                            'embeddings')
        # sentence plaholders have shape (batch, time_steps)
        self.sentence1 = tf.placeholder(tf.int32, (None, None), 'sentence1')
        self.sentence2 = tf.placeholder(tf.int32, (None, None), 'sentence2')
        self.sentence1_size = tf.placeholder(tf.int32, [None], 'sent1_size')
        self.sentence2_size = tf.placeholder(tf.int32, [None], 'sent2_size')
        self.label = tf.placeholder(tf.int32, [None], 'label')
        self.learning_rate = tf.placeholder(tf.float32, [],
                                            name='learning_rate')
        self.l2_constant = tf.placeholder(tf.float32, [], 'l2_constant')
        self.clip_value = tf.placeholder(tf.float32, [], 'clip_norm')
        self.dropout_keep = tf.placeholder(tf.float32, [], 'dropout')
        self.embedding_size = embedding_size

        # we initialize the embeddings from a placeholder to circumvent
        # tensorflow's limitation of 2 GB nodes in the graph
        self.embeddings = tf.Variable(self.embeddings_ph, trainable=False,
                                      validate_shape=True)

        # clip the sentences to the length of the longest one in the batch
        # this saves processing time
        clipped_sent1 = clip_sentence(self.sentence1, self.sentence1_size)
        clipped_sent2 = clip_sentence(self.sentence2, self.sentence2_size)
        embedded1 = tf.nn.embedding_lookup(self.embeddings, clipped_sent1)
        embedded2 = tf.nn.embedding_lookup(self.embeddings, clipped_sent2)
        repr1 = self._transformation_input(embedded1)
        repr2 = self._transformation_input(embedded2, True)

        # the architecture has 3 main steps: soft align, compare and aggregate
        # alpha and beta have shape (batch, time_steps, embeddings)
        self.alpha, self.beta = self.attend(repr1, repr2)
        self.v1 = self.compare(repr1, self.beta, self.sentence1_size)
        self.v2 = self.compare(repr2, self.alpha, self.sentence2_size, True)
        self.logits = self.aggregate(self.v1, self.v2)
        self.answer = tf.argmax(self.logits, 1, 'answer')

        hits = tf.equal(tf.cast(self.answer, tf.int32), self.label)
        self.accuracy = tf.reduce_mean(tf.cast(hits, tf.float32),
                                       name='accuracy')
        cross_entropy = tf.nn.\
            sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                     labels=self.label)
        self.labeled_loss = tf.reduce_mean(cross_entropy)
        weights = [v for v in tf.trainable_variables()
                   if 'weight' in v.name]
        l2_partial_sum = sum([tf.nn.l2_loss(weight) for weight in weights])
        l2_loss = tf.multiply(self.l2_constant, l2_partial_sum, 'l2_loss')
        self.loss = tf.add(self.labeled_loss, l2_loss, 'loss')

        if training:
            self._create_training_tensors(optimizer)

    def _transformation_input(self, inputs, reuse_weights=False):
        """
        Apply any transformations to the input embeddings

        :param inputs: a tensor with shape (batch, time_steps, embeddings)
        :return: a tensor of the same shape of the input
        """
        if self.project_input:
            projected = self.project_embeddings(inputs, reuse_weights)
            self.representation_size = self.num_units
        else:
            projected = inputs
            self.representation_size = self.embedding_size

        return projected

    def _create_training_tensors(self, optimizer_algorithm):
        """
        Create the tensors used for training
        """
        with tf.name_scope('training'):
            if optimizer_algorithm == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            elif optimizer_algorithm == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif optimizer_algorithm == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            else:
                ValueError('Unknown optimizer: %s' % optimizer_algorithm)

            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            if self.clip_value is not None:
                gradients, _ = tf.clip_by_global_norm(gradients,
                                                      self.clip_value)
            self.train_op = optimizer.apply_gradients(zip(gradients, v))

    def project_embeddings(self, embeddings, reuse_weights=False):
        """
        Project word embeddings into another dimensionality

        :param embeddings: embedded sentence, shape (batch, time_steps,
            embedding_size)
        :param reuse_weights: reuse weights in internal layers
        :return: projected embeddings with shape (batch, time_steps, num_units)
        """
        time_steps = tf.shape(embeddings)[1]
        embeddings_2d = tf.reshape(embeddings, [-1, self.embedding_size])

        with tf.variable_scope('projection', reuse=reuse_weights):
            initializer = tf.random_normal_initializer(0.0, 0.1)
            weights = tf.get_variable('weights',
                                      [self.embedding_size, self.num_units],
                                      initializer=initializer)

            projected = tf.matmul(embeddings_2d, weights)

        projected_3d = tf.reshape(projected,
                                  tf.stack([-1, time_steps, self.num_units]))
        return projected_3d

    def _transformation_compare(self, sentence, num_units, length,
                                reuse_weights=False):
        raise NotImplementedError()

    def _transformation_attend(self, sentence, num_units, length,
                               reuse_weights=False):
        raise NotImplementedError()

    def _apply_feedforward(self, inputs, scope,
                           reuse_weights=False, initializer=None,
                           num_units=None):
        """
        Apply two feed forward layers with self.num_units on the inputs.
        :param inputs: tensor in shape (batch, time_steps, num_input_units)
            or (batch, num_units)
        :param reuse_weights: reuse the weights inside the same tensorflow
            variable scope
        :param initializer: tensorflow initializer; by default a normal
            distribution
        :param num_units: list of length 2 containing the number of units to be
            used in each layer
        :return: a tensor with shape (batch, time_steps, num_units)
        """
        if num_units is None:
            num_units = [self.num_units, self.num_units]

        scope = scope or 'feedforward'
        with tf.variable_scope(scope, reuse=reuse_weights):
            with tf.variable_scope('layer1'):
                inputs = tf.nn.dropout(inputs, self.dropout_keep)
                relus = tf.layers.dense(inputs, num_units[0], tf.nn.relu,
                                        kernel_initializer=initializer)

            with tf.variable_scope('layer2'):
                inputs = tf.nn.dropout(relus, self.dropout_keep)
                relus2 = tf.layers.dense(inputs, num_units[1], tf.nn.relu,
                                         kernel_initializer=initializer)

        return relus2

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
        v1_sum = tf.reduce_sum(v1, 1)
        v2_sum = tf.reduce_sum(v2, 1)
        v1_max = tf.reduce_max(v1, 1)
        v2_max = tf.reduce_max(v2, 1)

        return tf.concat(axis=1, values=[v1_sum, v2_sum, v1_max, v2_max])

    def attend(self, sent1, sent2):
        """
        Compute inter-sentence attention. This is step 1 (attend) in the paper

        :param sent1: tensor in shape (batch, time_steps, num_units),
            the projected sentence 1
        :param sent2: tensor in shape (batch, time_steps, num_units)
        :return: a tuple of 3-d tensors, alfa and beta.
        """
        with tf.variable_scope('inter-attention') as self.attend_scope:
            # this is F in the paper
            num_units = self.representation_size

            # repr1 has shape (batch, time_steps, num_units)
            # repr2 has shape (batch, num_units, time_steps)
            repr1 = self._transformation_attend(sent1, num_units,
                                                self.sentence1_size)
            repr2 = self._transformation_attend(sent2, num_units,
                                                self.sentence2_size, True)
            repr2 = tf.transpose(repr2, [0, 2, 1])

            # compute the unnormalized attention for all word pairs
            # raw_attentions has shape (batch, time_steps1, time_steps2)
            self.raw_attentions = tf.matmul(repr1, repr2)

            # now get the attention softmaxes
            masked = mask_3d(self.raw_attentions, self.sentence2_size, -np.inf)
            att_sent1 = attention_softmax3d(masked)

            att_transposed = tf.transpose(self.raw_attentions, [0, 2, 1])
            masked = mask_3d(att_transposed, self.sentence1_size, -np.inf)
            att_sent2 = attention_softmax3d(masked)

            self.inter_att1 = att_sent1
            self.inter_att2 = att_sent2
            alpha = tf.matmul(att_sent2, sent1, name='alpha')
            beta = tf.matmul(att_sent1, sent2, name='beta')

        return alpha, beta

    def compare(self, sentence, soft_alignment, sentence_length,
                reuse_weights=False):
        """
        Apply a feed forward network to compare one sentence to its
        soft alignment with the other.

        :param sentence: embedded and projected sentence,
            shape (batch, time_steps, num_units)
        :param soft_alignment: tensor with shape (batch, time_steps, num_units)
        :param reuse_weights: whether to reuse weights in the internal layers
        :return: a tensor (batch, time_steps, num_units)
        """
        with tf.variable_scope('comparison', reuse=reuse_weights) \
                as self.compare_scope:
            num_units = 2 * self.representation_size

            # sent_and_alignment has shape (batch, time_steps, num_units)
            inputs = [sentence, soft_alignment, sentence - soft_alignment,
                      sentence * soft_alignment]
            sent_and_alignment = tf.concat(axis=2, values=inputs)

            output = self._transformation_compare(
                sent_and_alignment, num_units, sentence_length, reuse_weights)

        return output

    def aggregate(self, v1, v2):
        """
        Aggregate the representations induced from both sentences and their
        representations

        :param v1: tensor with shape (batch, time_steps, num_units)
        :param v2: tensor with shape (batch, time_steps, num_units)
        :return: logits over classes, shape (batch, num_classes)
        """
        inputs = self._create_aggregate_input(v1, v2)
        with tf.variable_scope('aggregation') as self.aggregate_scope:
            initializer = tf.random_normal_initializer(0.0, 0.1)
            with tf.variable_scope('linear'):
                shape = [self.num_units, self.num_classes]
                weights_linear = tf.get_variable('weights', shape,
                                                 initializer=initializer)
                bias_linear = tf.get_variable('bias', [self.num_classes],
                                              initializer=tf.zeros_initializer())

            pre_logits = self._apply_feedforward(inputs,
                                                 self.aggregate_scope)
            logits = tf.nn.xw_plus_b(pre_logits, weights_linear, bias_linear)

        return logits

    def initialize_embeddings(self, session, embeddings):
        """
        Initialize word embeddings
        :param session: tensorflow session
        :param embeddings: the contents of the word embeddings
        :return:
        """
        init_op = tf.variables_initializer([self.embeddings])
        session.run(init_op, {self.embeddings_ph: embeddings})

    def initialize(self, session, embeddings):
        """
        Initialize all tensorflow variables.
        :param session: tensorflow session
        :param embeddings: the contents of the word embeddings
        """
        init_op = tf.global_variables_initializer()
        session.run(init_op, {self.embeddings_ph: embeddings})

    @classmethod
    def _init_from_load(cls, params, training):
        """
        Call the constructor inside the loader
        :return: an instance of this class
        """
        return cls(params['num_units'], params['num_classes'],
                   params['vocab_size'], params['embedding_size'],
                   project_input=params['project_input'], training=training)

    @classmethod
    def load(cls, dirname, session, training=False):
        """
        Load a previously saved file.

        :param dirname: directory with model files
        :param session: tensorflow session
        :param training: whether to create training tensors
        :return: an instance of MultiFeedForward
        :rtype: DecomposableNLIModel
        """
        params = utils.load_parameters(dirname)
        model = cls._init_from_load(params, training)

        tensorflow_file = os.path.join(dirname, 'model')
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(session, tensorflow_file)

        # if training, optimizer values still have to be initialized
        if training:
            train_vars = [v for v in tf.global_variables()
                          if v.name.startswith('training')]
            init_op = tf.variables_initializer(train_vars)
            session.run(init_op)

        return model

    def _get_params_to_save(self):
        """
        Return a dictionary with data for reconstructing a persisted object
        """
        vocab_size = self.embeddings.get_shape()[0].value
        data = {'num_units': self.num_units,
                'num_classes': self.num_classes,
                'vocab_size': vocab_size,
                'embedding_size': self.embedding_size,
                'project_input': self.project_input}

        return data

    def save(self, dirname, session, saver):
        """
        Persist a model's information
        """
        params = self._get_params_to_save()
        tensorflow_file = os.path.join(dirname, 'model')
        params_file = os.path.join(dirname, 'model-params.json')

        with open(params_file, 'wb') as f:
            json.dump(params, f)

        saver.save(session, tensorflow_file)

    def _create_batch_feed(self, batch_data, learning_rate, dropout_keep,
                           l2, clip_value):
        """
        Create a feed dictionary to be given to the tensorflow session.
        """
        feeds = {self.sentence1: batch_data.sentences1,
                 self.sentence2: batch_data.sentences2,
                 self.sentence1_size: batch_data.sizes1,
                 self.sentence2_size: batch_data.sizes2,
                 self.label: batch_data.labels,
                 self.learning_rate: learning_rate,
                 self.dropout_keep: dropout_keep,
                 self.l2_constant: l2,
                 self.clip_value: clip_value
                 }
        return feeds

    def _run_on_validation(self, session, feeds):
        """
        Run the model with validation data, providing any useful information.

        :return: a tuple (validation_loss, validation_acc)
        """
        loss, acc = session.run([self.loss, self.accuracy], feeds)
        return loss, acc

    def train(self, session, train_dataset, valid_dataset, save_dir,
              learning_rate, num_epochs, batch_size, dropout_keep=1, l2=0,
              clip_norm=10, report_interval=1000):
        """
        Train the model

        :param session: tensorflow session
        :type train_dataset: utils.RTEDataset
        :type valid_dataset: utils.RTEDataset
        :param save_dir: path to a directory to save model files
        :param learning_rate: the initial learning rate
        :param num_epochs: how many epochs to train the model for
        :param batch_size: size of each batch
        :param dropout_keep: probability of keeping units during the dropout
        :param l2: l2 loss coefficient
        :param clip_norm: global norm to clip gradient tensors
        :param report_interval: number of batches before each performance
            report
        """
        logger = utils.get_logger(self.__class__.__name__)

        # this tracks the accumulated loss in a minibatch
        # (to take the average later)
        accumulated_loss = 0
        accumulated_accuracy = 0
        accumulated_num_items = 0

        best_acc = 0

        # batch counter doesn't reset after each epoch
        batch_counter = 0

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)

        for i in range(num_epochs):
            train_dataset.shuffle_data()
            batch_index = 0

            while batch_index < train_dataset.num_items:
                batch_index2 = batch_index + batch_size
                batch = train_dataset.get_batch(batch_index, batch_index2)
                feeds = self._create_batch_feed(batch, learning_rate,
                                                dropout_keep, l2, clip_norm)

                ops = [self.train_op, self.loss, self.accuracy]
                _, loss, accuracy = session.run(ops, feed_dict=feeds)
                accumulated_loss += loss * batch.num_items
                accumulated_accuracy += accuracy * batch.num_items
                accumulated_num_items += batch.num_items

                batch_index = batch_index2
                batch_counter += 1

                if batch_counter % report_interval == 0:
                    avg_loss = accumulated_loss / accumulated_num_items
                    avg_accuracy = accumulated_accuracy / accumulated_num_items
                    accumulated_loss = 0
                    accumulated_accuracy = 0
                    accumulated_num_items = 0

                    feeds = self._create_batch_feed(valid_dataset,
                                                    0, 1, l2, 0)

                    valid_loss, valid_acc = self._run_on_validation(session,
                                                                    feeds)

                    msg = '%d completed epochs, %d batches' % (i, batch_counter)
                    msg += '\tAvg train loss: %f' % avg_loss
                    msg += '\tAvg train acc: %.4f' % avg_accuracy
                    msg += '\tValidation loss: %f' % valid_loss
                    msg += '\tValidation acc: %.4f' % valid_acc

                    if valid_acc > best_acc:
                        best_acc = valid_acc
                        self.save(save_dir, session, saver)
                        msg += '\t(saved model)'

                    logger.info(msg)

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
        while i < dataset.num_items:
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
