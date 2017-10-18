import tensorflow as tf
import numpy as np

import logging
import collections
import os


logger = logging.getLogger(__name__)


RunContext = collections.namedtuple(
    'RunContext',
    ['logits_op', 'train_op', 'loss_op', 'acc_op', 'steps_per_epoch']
)


def run_training_epoch(sess, context):
    epoch_loss, epoch_accs = [], []
    for i in range(context.steps_per_epoch):
        _, loss, acc = sess.run(
            [context.train_op, context.loss_op, context.acc_op]
        )
        epoch_loss.append(loss)
        epoch_accs.append(acc)

    return np.mean(epoch_loss), np.mean(epoch_accs)


def eval_epoch(sess, context):
    losses, accs = [], []
    for _ in range(context.steps_per_epoch):
        loss, acc = sess.run([context.loss_op, context.acc_op])
        losses.append(loss)
        accs.append(acc)

    return np.mean(losses), np.mean(accs)


def progress(strip):
    """
    As detailed in:
        Early stopping - but when?. Lutz Prechelt (1997)

    Progress is the measure of how much training error during a strip
    is larger than the minimum training error during the strip.
    """
    k = len(strip)
    return 1000 * ((np.sum(strip)/(np.min(strip)*k)) - 1)


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
