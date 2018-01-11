import os
import logging

import tensorflow as tf

from layout import get_layer_id
from kernels import KERNEL_ASSIGN_OPS


logger = logging.getLogger(__name__)


def save_model(monitored_sess, saver, folder, step):
    path = os.path.join(folder, 'model_' + str(step) + '.ckpt')
    sess = monitored_sess._sess._sess._sess._sess
    saver.save(sess, path)
    return path


def init_kernel_ops(sess):
    sess.run(tf.get_collection(KERNEL_ASSIGN_OPS))


def create_global_step():
    """ Creates a global step in the VARIABLEs and GLOBAL_STEP collections """
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP]
    return tf.get_variable('global_step', shape=[],
                           dtype=tf.int32,
                           initializer=tf.constant_initializer(0),
                           trainable=False,
                           collections=collections)


def get_accuracy_op(logits, labels, n_classes):
    if n_classes == 2:
        # Labels should be either 0 or 1
        predicted = tf.squeeze(
            _binary_activation(tf.nn.sigmoid(logits)),
            1
        )
    else:
        predicted = tf.argmax(tf.nn.softmax(logits), 1)

    casted_labels = tf.squeeze(tf.cast(labels, tf.int64), 1)
    correct_pred = tf.equal(predicted, casted_labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def _binary_activation(x):
    negative_idx = tf.less(x, tf.ones(tf.shape(x)) * 0.5)
    zero_tensor = tf.zeros(tf.shape(x))
    one_tensor = tf.ones(tf.shape(x))
    return tf.cast(tf.where(negative_idx, zero_tensor, one_tensor), tf.int64)


def get_model_weights(layers, include_output=True):
    """ Returns the list of models parameters """
    weights = []
    vars_layers = variables_from_layers(layers, include_output)
    for var in vars_layers:
        if 'bias' in var.name:
            logger.debug('Ignoring bias %s for regularization ' % (var.name))
        elif 'weight' in var.name:
            weights.append(var)
            logger.debug('Using weights %s for regularization ' % (var.name))
        else:
            logger.warning('Ignoring unknown parameter type %s' % (var.name))
    return weights


def variables_from_layers(layer_list, include_output=True):
    selected = []
    train_vars = tf.get_collection(tf.GraphKeys.WEIGHTS)
    for var in train_vars:
        try:
            layer_name = var.name.split('/')[1]
            layer_id = get_layer_id(layer_name)
            if int(layer_id) in layer_list:
                selected.append(var)
        except Exception:
            # If we are here we assume we have an output variable
            if include_output:
                selected.append(var)
    return selected


def get_loss_fn(logits, labels, n_classes):
    if n_classes == 2:
        return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.cast(labels, tf.float32), logits=logits
        )
    elif n_classes > 2:
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(labels, depth=n_classes),
            logits=logits
        )
    else:
        raise ValueError('Number of outputs must be at least 2')


def get_loss_op(logits, y, weights, sum_collection, n_classes, **params):
    l2_ratio = params.get('l2_ratio', None)

    l2_term = l2_norm(weights) * l2_ratio \
        if l2_ratio is not None else tf.constant(0.0)

    loss_term = tf.reduce_mean(
        get_loss_fn(
            logits, y, n_classes
        )
    )

    loss_op = loss_term + l2_term
    return loss_op


def loss_ops_list(logits, y, sum_collection, n_classes, num_layers, **params):
    """
    Builds a tensor with loss ops where the ith position
    corresponds to the operation to train layer i. The zero position
    is the function where we all layers
    """
    loss_ops = []

    # First position is for all layers
    all_weights = get_model_weights(list(range(1, num_layers + 1)))
    loss_all = get_loss_op(
        logits,
        y,
        all_weights,
        sum_collection,
        n_classes,
        **params
    )
    loss_ops.append(loss_all)
    logger.debug('L2 #{} uses {}'.format(0, all_weights))

    for i in range(1, num_layers + 1):
        layer_weights = get_model_weights([i])
        layer_loss = get_loss_op(
            logits,
            y,
            layer_weights,
            sum_collection,
            n_classes
        )
        logger.debug('L2 #{} uses {}'.format(i, layer_weights))
        loss_ops.append(layer_loss)

    return loss_ops


def train_ops_list(lr, loss_ops, n_layers):
    """
    Builds a tensor with training ops where the ith position
    corresponds to the operation to train layer i. The zero position
    is the function where we optimize everything
    """
    train_ops = []

    # First position is for all
    train_ops.append(
        get_train_op(lr, loss_ops[0], tf.trainable_variables())
    )
    logger.debug('Optimizer #{} uses {}'.format(0, tf.trainable_variables()))

    for i in range(1, n_layers + 1):
        opt_vars = variables_from_layers([i], True)
        logger.debug('Optimizer #{} uses {}'.format(i, opt_vars))
        train_ops.append(
            get_train_op(lr, loss_ops[i], opt_vars)
        )

    return train_ops


def get_train_op(lr, loss_op, opt_var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # This is just a safe option if we use update ops such
    # as moving averages (e.g. batch norm)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(
            loss_op, var_list=opt_var_list,
        )

    return train_op


def l2_norm(weights):
    return tf.add_n([tf.nn.l2_loss(x) for x in weights])


def get_l2_ops_list(**params):

    l2_ratio = params.get('l2_ratio', None)
    num_layers = params.get('num_layers')

    def get_l2_op(weights):
        return l2_norm(weights) * l2_ratio \
            if l2_ratio is not None else tf.constant(0.0)

    l2_list = []
    all_weights = variables_from_layers(range(1, num_layers+1))
    l2_list.append(get_l2_op(all_weights))

    for i in range(1, num_layers+1):
        current_weights = variables_from_layers([i])
        l2_list.append(get_l2_op(current_weights))

    return l2_list
