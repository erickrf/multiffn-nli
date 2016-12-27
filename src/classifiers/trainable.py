# -*- coding: utf-8 -*-

import abc
import tensorflow as tf

import utils


class Trainable(object):
    """
    Abstract class for trainable model
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.train_op = None
        self.loss = None
        raise NotImplementedError('Abstract class')

    def save(self, save_dir, session, saver):
        pass

    def _create_batch_feed(self, sentence1, sentence2, size1, size2,
                           label, learning_rate, dropout_keep, l2, clip_value):
        """
        Create a feed dictionary to be given to the tensorflow session.
        """
        pass

    def _run_on_validation(self, session, feeds):
        """
        Run the model with validation data, providing any useful information.
        :return: a tuple (validation_loss, validation_msg)
            validation_msg should include information to be displayed about
            validation performance
        """
        pass

    def _train(self, session, vars_to_save, save_dir, train_dataset,
               valid_dataset, learning_rate, num_epochs, batch_size,
               dropout_keep=1, l2=0, clip_norm=-1, report_interval=1000):
        logger = utils.get_logger(self.__class__.__name__)

        # this tracks the accumulated loss in a minibatch (to take the average later)
        accumulated_loss = 0

        best_loss = 10e10

        # batch counter doesn't reset after each epoch
        batch_counter = 0

        saver = tf.train.Saver(vars_to_save, max_to_keep=1)
        # summ_writer = tf.train.SummaryWriter(log_dir, session.graph)
        # summ_writer.add_graph(session.graph)

        for i in range(num_epochs):
            train_dataset.shuffle_data()
            batch_index = 0

            while batch_index < train_dataset.num_items:
                batch_index2 = batch_index + batch_size
                sent1 = train_dataset.sentences1[batch_index:batch_index2]
                sent2 = train_dataset.sentences2[batch_index:batch_index2]
                size1 = train_dataset.sizes1[batch_index:batch_index2]
                size2 = train_dataset.sizes2[batch_index:batch_index2]
                label = train_dataset.labels[batch_index:batch_index2]

                feeds = self._create_batch_feed(sent1, sent2, size1, size2,
                                                label, learning_rate,
                                                dropout_keep, l2, clip_norm)

                ops = [self.train_op, self.loss]
                _, loss = session.run(ops, feed_dict=feeds)
                accumulated_loss += loss

                batch_index = batch_index2
                batch_counter += 1
                if batch_counter % report_interval == 0:
                    # summ_writer.add_summary(summaries, batch_counter)
                    avg_loss = accumulated_loss / report_interval
                    accumulated_loss = 0

                    feeds = self._create_batch_feed(valid_dataset.sentences1,
                                                    valid_dataset.sentences2,
                                                    valid_dataset.sizes1,
                                                    valid_dataset.sizes2,
                                                    valid_dataset.labels,
                                                    0, 1, l2, 0)

                    valid_loss, valid_msg = self._run_on_validation(session, feeds)

                    msg = '%d completed epochs, %d batches' % (i, batch_counter)
                    msg += '\tAverage training batch loss: %f' % avg_loss
                    msg += '\t' + valid_msg

                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        self.save(save_dir, session, saver)
                        msg += '\t(saved model)'

                    logger.info(msg)


def get_weights_and_biases():
    """
    Return all weight and bias variables
    :return:
    """
    return [var for var in tf.global_variables()
            if 'weight' in var.name or 'bias' in var.name]
