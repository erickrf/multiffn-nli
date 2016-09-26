# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
import numpy as np
import json
import os

import utils


def variable_summaries(var, name):
    """
    Create tensorflow variable summaries
    """
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def attention_softmax3d(values):
    """
    Performs a softmax over the attention values.

    :param values: 3d tensor with raw values
    :return: 3d tensor, same shape as input
    """
    original_shape = tf.shape(values)
    num_units = original_shape[2]
    reshaped_values = tf.reshape(values, tf.pack([-1, num_units]))
    softmaxes = tf.nn.softmax(reshaped_values)
    return tf.reshape(softmaxes, original_shape)


def clip_sentence(sentence, sizes):
    """
    Clip the input sentence placeholders to the length of the longest one in the
    batch. This saves processing time.

    :param sentence: tensor with shape (time_steps, batch)
    :param sizes: tensor with shape (batch)
    :return: tensor with shape (time_steps, batch)
    """
    max_batch_size = tf.reduce_max(sizes)
    clipped_sent = tf.slice(sentence, [0, 0],
                            tf.pack([max_batch_size, -1]))
    return clipped_sent


def mask_values_after_sentence_end(values, sentence_sizes, value):
    """
    Given a batch of matrices, each with shape m x n, mask the values in each row
    after the positions indicated in sentence_sizes.

    :param values: tensor with shape (batch_size, m, n)
    :param sentence_sizes: tensor with shape (batch_size) containing the
        sentence sizes that should be limited
    :param value: scalar value to assign to items after sentence size
    :return: a tensor with the same shape
    """
    time_steps = tf.shape(values)[2]

    ones = tf.ones_like(values, dtype=tf.int32)
    mask = value * tf.cast(ones, tf.float32)

    # This piece of code is pretty ugly. We create a tensor with the same shape
    # as the values with each index from 0 to max_size and compare it against
    # another tensor with the same shape which holds the length of each batch.
    # We use tf.select, and then set values past sentence size to -inf.
    # If/when tensorflow had better indexing capabilities, we could simplify it.
    range_ = tf.range(time_steps)
    positions = ones * tf.reshape(range_, [1, 1, -1])
    sizes = ones * tf.reshape(sentence_sizes, [-1, 1, 1])
    cond = tf.less(positions, sizes)

    return tf.select(cond, values, mask)


class MultiFeedForward(object):
    """
    Implementation of the multi feed forward network model described in
    the paper "A Decomposable Attention Model for Natural Language
    Inference" by Parikh et al., 2016.

    It applies feedforward MLPs to combinations of parts of the two sentences,
    without any recurrent structure.
    """
    def __init__(self, num_units, max_size1, max_size2, num_classes, embedding_size,
                 use_intra_attention=False, training=True, learning_rate=0.001,
                 clip_value=None, l2_constant=0.0, distance_biases=10):

        self.max_time_steps1 = max_size1
        self.max_time_steps2 = max_size2
        self.num_units = num_units
        self.num_classes = num_classes
        self.use_intra = use_intra_attention
        self.distance_biases = distance_biases
        self.embeddings_ph = tf.placeholder(tf.float32, (None, embedding_size), 'embeddings')
        self.sentence1 = tf.placeholder(tf.int32, (max_size1, None), 'sentence1')
        self.sentence2 = tf.placeholder(tf.int32, (max_size2, None), 'sentence2')
        self.sentence1_size = tf.placeholder(tf.int32, [None], 'sent1_size')
        self.sentence2_size = tf.placeholder(tf.int32, [None], 'sent2_size')
        self.label = tf.placeholder(tf.int32, [None], 'label')
        self.l2_constant = l2_constant
        self.learning_rate = learning_rate
        self.clip_value = clip_value
        self.dropout_keep = tf.placeholder(tf.float32, None, 'dropout')
        self.embedding_size = embedding_size

        # we initialize the embeddings from a placeholder to circumvent
        # tensorflow's limitation of 2 GB nodes in the graph
        self.embeddings = tf.Variable(self.embeddings_ph, trainable=False,
                                      validate_shape=False)

        # clip the sentences to the length of the longest one in the batch
        # this saves processing time
        clipped_sent1 = clip_sentence(self.sentence1, self.sentence1_size)
        clipped_sent2 = clip_sentence(self.sentence2, self.sentence2_size)
        embedded1 = tf.nn.embedding_lookup(self.embeddings, clipped_sent1)
        embedded2 = tf.nn.embedding_lookup(self.embeddings, clipped_sent2)
        projected1 = self.project_embeddings(embedded1)
        projected2 = self.project_embeddings(embedded2, True)

        # the architecture has 3 main steps: soft align, compare and aggregate

        # alpha and beta have shape (batch, time_steps, embeddings)
        if use_intra_attention:
            repr1 = self.compute_intra_attention(projected1)
            repr2 = self.compute_intra_attention(projected2, True)
        else:
            repr1 = projected1
            repr2 = projected2

        self.alpha, self.beta = self.attend(repr1, repr2)
        self.v1 = self.compare(repr1, self.beta)
        self.v2 = self.compare(repr2, self.alpha, True)
        self.logits = self.aggregate(self.v1, self.v2)
        self.answer = tf.argmax(self.logits, 1, 'answer')

        if training:
            self._create_training_tensors()
            self.merged_summaries = tf.merge_all_summaries()

    def project_embeddings(self, embeddings, reuse_weights=False):
        """
        Project word embeddings into another dimensionality

        :param embeddings: embedded sentence, shape (time_steps, batch, embedding_size)
        :param reuse_weights: reuse weights in internal layers
        :return: projected embeddings with shape (batch, time_steps, num_units)
        """
        time_steps = tf.shape(embeddings)[0]
        embeddings_2d = tf.reshape(embeddings, [-1, self.embedding_size])

        with tf.variable_scope('projection', reuse=reuse_weights):
            initializer = tf.random_normal_initializer(0.0, 0.1)
            weights = tf.get_variable('weights', [self.embedding_size, self.num_units],
                                      initializer=initializer)
            if not reuse_weights:
                variable_summaries(weights, 'projection/weights')

            projected = tf.matmul(embeddings_2d, weights)

        projected_3d = tf.reshape(projected, tf.pack([time_steps, -1, self.num_units]))
        projected_3d_batch_first = tf.transpose(projected_3d, [1, 0, 2])
        return projected_3d_batch_first

    def _relu_layer(self, inputs, weights, bias):
        """
        Apply dropout to the inputs, followed by the weights and bias,
        and finally the relu activation
        :param inputs: 2d tensor
        :param weights: 2d tensor
        :param bias: 1d tensor
        :return: 2d tensor
        """
        after_dropout = tf.nn.dropout(inputs, self.dropout_keep)
        raw_values = tf.nn.xw_plus_b(after_dropout, weights, bias)
        return tf.nn.relu(raw_values)

    def _apply_network(self, inputs, num_input_units, reuse_weights=False):
        """
        Apply two feed forward layers with self.num_units on the inputs.
        :param inputs: tensor in shape (batch, time_steps, num_input_units)
            or (batch, num_units)
        :param num_input_units: a python int
        :return: a tensor with shape (batch, time_steps, num_units)
        """
        rank = len(inputs.get_shape())
        if rank == 3:
            time_steps = tf.shape(inputs)[1]

            # combine batch and time steps in the first dimension
            inputs2d = tf.reshape(inputs, tf.pack([-1, num_input_units]))
        else:
            inputs2d = inputs

        with tf.variable_scope('feedforward', reuse=reuse_weights):
            initializer = tf.random_normal_initializer(0.0, 0.1)

            with tf.variable_scope('layer1'):
                shape = [num_input_units, self.num_units]
                weights1 = tf.get_variable('weights', shape, initializer=initializer)
                bias1 = tf.Variable(tf.zeros([self.num_units]), name='bias')

            with tf.variable_scope('layer2'):
                shape = [self.num_units, self.num_units]
                weights2 = tf.get_variable('weights', shape, initializer=initializer)
                bias2 = tf.Variable(tf.zeros([self.num_units]), name='bias')

            # relus are (time_steps * batch, num_units)
            relus1 = self._relu_layer(inputs2d, weights1, bias1)
            relus2 = self._relu_layer(relus1, weights2, bias2)

        if rank == 3:
            output_shape = tf.pack([-1, time_steps, self.num_units])
            return tf.reshape(relus2, output_shape)

        return relus2

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
            clipped_inds = tf.clip_by_value(raw_inds, 0, self.distance_biases - 1)
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
        with tf.variable_scope('intra-attention'):
            # this is F_intra in the paper
            # f_intra1 is (batch, time_steps, num_units) and
            # f_intra1_t is (batch, num_units, time_steps)
            f_intra = self._apply_network(sentence, self.num_units,
                                          reuse_weights=reuse_weights)
            f_intra_t = tf.transpose(f_intra, [0, 2, 1])

            # these are f_ij
            # raw_attentions is (batch, time_steps, time_steps)
            raw_attentions = tf.batch_matmul(f_intra, f_intra_t)

            # bias has shape (time_steps, time_steps)
            bias = self._get_distance_biases(time_steps, reuse_weights=reuse_weights)

            # bias is broadcast along batches
            raw_attentions += bias
            attentions = attention_softmax3d(raw_attentions)

            attended = tf.batch_matmul(attentions, sentence)

        return tf.concat(2, [sentence, attended])

    def attend(self, sent1, sent2):
        """
        Compute inter-sentence attention. This is step 1 (attend) in the paper

        :param sent1: tensor in shape (batch, time_steps, num_units),
            the projected sentence 1
        :param sent2: tensor in shape (batch, time_steps, num_units)
        :return: a tuple of 3-d tensors, alfa and beta.
        """
        with tf.variable_scope('inter-attention'):
            # this is F in the paper

            if self.use_intra:
                num_units = 2 * self.num_units
            else:
                num_units = self.num_units

            # repr1 has shape (batch, time_steps, num_units)
            # repr2 has shape (batch, num_units, time_steps)
            repr1 = self._apply_network(sent1, num_units)
            repr2 = self._apply_network(sent2, num_units, True)
            repr2 = tf.transpose(repr2, [0, 2, 1])

            # compute the unnormalized attention for all word pairs
            # raw_attentions has shape (batch, time_steps1, time_steps2)
            raw_attentions = tf.batch_matmul(repr1, repr2)

            # now get the attention softmaxes
            att_sent1 = attention_softmax3d(raw_attentions)

            att_transposed = tf.transpose(raw_attentions, [0, 2, 1])
            att_sent2 = attention_softmax3d(att_transposed)

            self.inter_att1 = att_sent1
            self.inter_att2 = att_sent2
            alpha = tf.batch_matmul(att_sent2, sent1, name='alpha')
            beta = tf.batch_matmul(att_sent1, sent2, name='beta')

        return alpha, beta

    def compare(self, sentence, soft_alignment, reuse_weights=False):
        """
        Apply a feed forward network to compare the one sentence to its
        soft alignment with the other.

        :param sentence: embedded and projected sentence,
            shape (batch, time_steps, num_units)
        :param soft_alignment: tensor with shape (batch, time_steps, num_units)
        :param reuse_weights: whether to reuse weights in the internal layers
        :return: a tensor (batch, time_steps, num_units)
        """
        with tf.variable_scope('comparison', reuse=reuse_weights):
            if self.use_intra:
                # sentence representation has 2*self.num_units, plus
                # 2*self.num_units from the soft alignments
                num_units = 4 * self.num_units
            else:
                num_units = 2 * self.num_units

            # sent_and_alignment has shape (batch, time_steps, num_units)
            sent_and_alignment = tf.concat(2, [sentence, soft_alignment])

            output = self._apply_network(sent_and_alignment, num_units)

        return output

    def aggregate(self, v1, v2):
        """
        Aggregate the representations induced from both sentences and their
        representations
        :param v1: tensor with shape (batch, time_steps, num_units)
        :param v2: tensor with shape (batch, time_steps, num_units)
        :return: logits over classes, shape (batch, num_classes)
        """
        # sum over time steps; resulting shape is (batch, num_units)
        v1_sum = tf.reduce_sum(v1, [1])
        v2_sum = tf.reduce_sum(v2, [1])
        concat_v = tf.concat(1, [v1_sum, v2_sum])

        with tf.variable_scope('aggregation'):
            initializer = tf.random_normal_initializer(0.0, 0.1)
            with tf.variable_scope('linear'):
                shape = [self.num_units, self.num_classes]
                weights_linear = tf.get_variable('weights', shape,
                                                 initializer=initializer)
                bias_linear = tf.get_variable('bias', [self.num_classes],
                                              initializer=tf.zeros_initializer)

            pre_logits = self._apply_network(concat_v, 2 * self.num_units)
            logits = tf.nn.xw_plus_b(pre_logits, weights_linear, bias_linear)

        return logits

    def _create_training_tensors(self):
        """
        Create the tensors used for training
        """
        hits = tf.equal(tf.cast(self.answer, tf.int32), self.label)
        self.accuracy = tf.reduce_mean(tf.cast(hits, tf.float32),
                                       name='accuracy')
        with tf.name_scope('training'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits,
                                                                           self.label)
            labeled_loss = tf.reduce_mean(cross_entropy)
            weights = [v for v in tf.all_variables() if 'weight' in v.name]
            if self.l2_constant > 0:
                l2_partial_sum = sum([tf.nn.l2_loss(weight) for weight in weights])
                l2_loss = tf.mul(self.l2_constant, l2_partial_sum, 'l2_loss')
                self.loss = tf.add(labeled_loss, l2_loss, 'loss')
            else:
                self.loss = labeled_loss

            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            if self.clip_value is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
            self.train_op = optimizer.apply_gradients(zip(gradients, v))

    def initialize(self, session, embeddings):
        """
        Initialize all tensorflow variables.
        :param session: tensorflow session
        :param embeddings: the contents of the word embeddings
        """
        init_op = tf.initialize_all_variables()
        session.run(init_op, {self.embeddings_ph: embeddings})

    @classmethod
    def load(cls, dirname, session):
        """
        Load a previously saved file.
        :param dirname: directory with model files
        :param session: tensorflow session
        :return: an instance of MultiFeedForward
        """
        params = utils.load_parameters(dirname)

        # create a tensor of zeros just to fill the graph where embeddings are expected
        # when tf.saver.load() is called, they will be replaced with the actual values
        embedding_placeholder = np.zeros(params['embedding_shape'], dtype=np.float32)
        model = cls(params['num_units'], params['time_steps1'], params['time_steps2'],
                    params['num_classes'], embedding_placeholder, training=False)

        tensorflow_file = os.path.join(dirname, 'model')
        saver = tf.train.Saver()
        saver.restore(session, tensorflow_file)

        return model

    def _get_params_to_save(self, session):
        """
        Return a dictionary with data for reconstructing a persisted object
        """
        embedding_shape_tf = tf.shape(self.embeddings)
        embedding_shape = session.run(embedding_shape_tf).tolist()

        data = {'time_steps1': self.max_time_steps1,
                'time_steps2': self.max_time_steps2,
                'num_units': self.num_units,
                'num_classes': self.num_classes,
                'embedding_shape': embedding_shape}

        return data

    def save(self, dirname, session, saver):
        """
        Persist a model's information
        """
        params = self._get_params_to_save(session)
        tensorflow_file = os.path.join(dirname, 'model')
        params_file = os.path.join(dirname, 'model-params.json')

        with open(params_file, 'wb') as f:
            json.dump(params, f)

        saver.save(session, tensorflow_file)

    def train(self, session, train_dataset, valid_dataset, embeddings,
              num_epochs, batch_size, dropout_keep, save_dir, log_dir,
              report_interval=100):
        """
        Train the model with the specified parameters
        :param session: tensorflow session
        :param train_dataset: an RTEDataset object with training data
        :param valid_dataset: an RTEDataset object with validation data
        :param num_epochs: number of epochs to run the model. During each epoch,
            all data points are seen exactly once
        :param batch_size: how many items in each minibatch.
        :param dropout_keep: dropout keep probability (applied at LSTM input and output)
        :param save_dir: path to directory to save the model
        :param log_dir: directory to save logs
        :param report_interval: how many minibatches between each performance report
        :return:
        """
        logger = utils.get_logger('rte_network')

        # this tracks the accumulated loss in a minibatch (to take the average later)
        accumulated_loss = 0

        best_acc = 0

        # batch counter doesn't reset after each epoch
        batch_counter = 0

        # get all weights and biases, but not the embeddings
        # (embeddings are huge and saved separately)
        vars_to_save = [var for var in tf.all_variables()
                        if 'weight' in var.name or 'bias' in var.name]

        saver = tf.train.Saver(vars_to_save, max_to_keep=1)
        summ_writer = tf.train.SummaryWriter(log_dir, session.graph)
        summ_writer.add_graph(session.graph)

        for i in range(num_epochs):
            train_dataset.shuffle_data()
            batch_index = 0

            while batch_index < train_dataset.num_items:
                batch_index2 = batch_index + batch_size

                # transpose arrays so that input is (num_time_steps, batch_size)
                feeds = {self.sentence1: train_dataset.sentences1[batch_index:batch_index2].T,
                         self.sentence2: train_dataset.sentences2[batch_index:batch_index2].T,
                         self.sentence1_size: train_dataset.sizes1[batch_index:batch_index2],
                         self.sentence2_size: train_dataset.sizes2[batch_index:batch_index2],
                         self.label: train_dataset.labels[batch_index:batch_index2],
                         self.dropout_keep: dropout_keep
                         }

                ops = [self.train_op, self.loss, self.merged_summaries]
                _, loss, summaries = session.run(ops, feed_dict=feeds)
                accumulated_loss += loss

                batch_index = batch_index2
                batch_counter += 1
                if batch_counter % report_interval == 0:
                    summ_writer.add_summary(summaries, batch_counter)
                    avg_loss = accumulated_loss / report_interval
                    accumulated_loss = 0

                    feeds = {self.sentence1: valid_dataset.sentences1.T,
                             self.sentence2: valid_dataset.sentences2.T,
                             self.sentence1_size: valid_dataset.sizes1,
                             self.sentence2_size: valid_dataset.sizes2,
                             self.label: valid_dataset.labels,
                             self.dropout_keep: 1.0
                             }

                    valid_loss, acc = session.run([self.loss, self.accuracy],
                                                  feed_dict=feeds)

                    msg = '%d completed epochs, %d batches' % (i, batch_counter)
                    msg += '\tAverage training batch loss: %f' % avg_loss
                    msg += '\tValidation loss: %f' % valid_loss
                    msg += '\tValidation accuracy: %f' % acc
                    if acc > best_acc:
                        best_acc = acc
                        self.save(save_dir, session, saver)
                        msg += '\t(saved model)'

                    logger.info(msg)

        summ_writer.flush()
        summ_writer.close()
