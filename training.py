import tensorflow as tf
import numpy as np

import logging
import os

logger = logging.getLogger(__name__)


def get_writer(graph, folder, data_mode):
    writer_path = os.path.join(folder, data_mode)
    os.makedirs(writer_path)
    return tf.summary.FileWriter(writer_path, graph)


def get_global_step(graph=None):
    """ Loads the unique global step, if found """
    graph = tf.get_default_graph() if graph is None else graph
    global_step_tensors = graph.get_collection(tf.GraphKeys.GLOBAL_STEP)
    if len(global_step_tensors) == 0:
        raise RuntimeError('No global step stored in the collection')
    elif len(global_step_tensors) > 1:
        raise RuntimeError('Multiple instances found for the global step')
    return global_step_tensors[0]


def create_global_step():
    """ Creates a global step in the VARIABLEs and GLOBAL_STEP collections """
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP]
    return tf.get_variable('global_step', shape=[],
                           dtype=tf.int32,
                           initializer=tf.constant_initializer(0),
                           trainable=False,
                           collections=collections)


def l1_norm(weights):
    return tf.add_n([tf.reduce_sum(tf.abs(x)) for x in weights])


def l2_norm(weights):
    l2_loss = tf.add_n([tf.nn.l2_loss(x) for x in weights])
    return tf.divide(l2_loss, 2.0)


def get_model_weights():
    """ Returns the list of models parameters """
    weights = []
    for var in tf.get_collection(tf.GraphKeys.WEIGHTS):
        if 'bias' in var.name:
            logger.info('Ignoring bias %s for regularization ' % (var.name))
        elif 'weight' in var.name:
            weights.append(var)
            logger.info('Using weights %s for regularization ' % (var.name))
        else:
            logger.info('Ignoring unknown parameter type %s' % (var.name))
    return weights


def get_accuracy(softmax, labels):
    predicted = tf.argmax(softmax, 1)
    casted_labels = tf.squeeze(tf.cast(labels, tf.int64), 1)
    correct_pred = tf.equal(predicted, casted_labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def save_model(monitored_sess, saver, folder, step):
    path = os.path.join(folder, 'model_' + str(step) + '.ckpt')
    sess = monitored_sess._sess._sess._sess._sess
    saver.save(sess, path)
    return path


def eval_epoch(sess, loss_op, acc_op, steps):
    losses, accs = [], []
    for _ in range(steps):
        loss, acc = sess.run([loss_op, acc_op])
        losses.append(loss)
        accs.append(acc)

    mean_loss = np.mean(losses)
    mean_acc = np.mean(accs)
    return mean_loss, mean_acc
