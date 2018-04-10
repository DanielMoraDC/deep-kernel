import os
import logging

import tensorflow as tf

from kernels import KERNEL_ASSIGN_OPS, GaussianRFF, sample_w, \
                    is_w, kernel_dropout_w
from variables import get_model_weights, get_trainable_params, \
                    summarize_gradients, get_kernel_vars, get_variable_name

logger = logging.getLogger(__name__)


def save_model(monitored_sess, saver, folder, step):
    path = os.path.join(folder, 'model_' + str(step) + '.ckpt')
    sess = monitored_sess._sess._sess._sess._sess
    saver.save(sess, path)
    return path


def init_kernel_ops(sess):
    sess.run(tf.get_collection(KERNEL_ASSIGN_OPS))


def get_global_step():
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


def get_l2_op(weights, **params):
    l2_ratio = params.get('l2_ratio', None)
    return l2_norm(weights) * l2_ratio \
        if l2_ratio is not None else tf.constant(0.0)


def get_loss_op(logits, y, weights, sum_collection, n_classes, **params):
    loss_term = tf.reduce_mean(
        get_loss_fn(
            logits, y, n_classes
        )
    )
    loss_op = loss_term + get_l2_op(weights, **params)
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
            n_classes,
            **params
        )
        logger.debug('L2 #{} uses {}'.format(i, layer_weights))
        loss_ops.append(layer_loss)

    return loss_ops


def train_ops_list(lr, loss_ops, n_layers, tag):
    """
    Builds a tensor with training ops where the ith position
    corresponds to the operation to train layer i. The zero position
    is the function where we optimize everything
    """
    train_ops = []

    # First position is for all
    train_ops.append(
        get_train_op(lr, loss_ops[0], tf.trainable_variables(), tag)
    )
    logger.debug('Optimizer #{} uses {}'.format(0, tf.trainable_variables()))

    for i in range(1, n_layers + 1):
        opt_vars = get_trainable_params([i], True)
        logger.debug('Optimizer #{} uses {}'.format(i, opt_vars))
        train_ops.append(
            get_train_op(lr, loss_ops[i], opt_vars, tag)
        )

    return train_ops


def get_train_op(lr, loss_op, opt_var_list, tag):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # This is just a safe option if we use update ops such
    # as moving averages (e.g. batch norm)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grads = optimizer.compute_gradients(loss_op, var_list=opt_var_list)
        summarize_gradients(grads, tag)
        train_op = optimizer.apply_gradients(grads)

    return train_op


def l2_norm(weights):
    return tf.add_n([tf.nn.l2_loss(x) for x in weights])


def get_l2_ops_list(**params):
    num_layers = params.get('num_layers')

    l2_list = []
    all_weights = get_model_weights(range(1, num_layers+1))
    logger.debug('L2 stats vars #0: {}'.format(all_weights))
    l2_list.append(get_l2_op(all_weights, **params))

    for i in range(1, num_layers+1):
        current_weights = get_model_weights([i])
        logger.debug('L2 stats vars #{}: {}'.format(i, current_weights))
        l2_list.append(get_l2_op(current_weights, **params))

    return l2_list


def get_kernel_assign_ops(layers, kernel_dropout_rate, **params):

    if kernel_dropout_rate < 0.0 or kernel_dropout_rate > 1.0:
        raise ValueError('Persist ratio must be in interval [0,1]')

    ops = []
    kernel_vars = get_kernel_vars(layers, include_fc=True)
    for var in kernel_vars:

        if not is_w(var.name):
            continue

        # CNN layers and FC layers have different kernel feature number
        sample_size = params['cnn_kernel_size'] if 'cnn' in var.name \
            else params['kernel_size']
        sample_params = {
            "kernel_size": sample_size,
            "kernel_std": params['kernel_std'],
        }

        w_sample = sample_w(GaussianRFF, var, **sample_params)

        # Construct a new matrix with some rows updated (new RFF features)
        new_w = kernel_dropout_w(var, w_sample, kernel_dropout_rate)

        # Append assign operation for current layer kernel matrix
        var_name = var.name.split('/')[1]
        var_id = str(get_variable_name(var_name))
        assign_b_op = tf.assign(var, new_w, name='assign_%s' % var_id)
        ops.append(assign_b_op)

    return ops


def get_kernel_assign_ops_list(num_layers, **params):
    ops = []

    # For entry 0, we get all assign ops
    ops.append(
        get_kernel_assign_ops(range(1, num_layers+1), **params)
    )
    logger.debug('Kernel ops for #0: {}'.format(ops[0]))

    # Then we get assign ops for each separate layer
    for l in range(1, num_layers+1):
        layer_kernel_ops = get_kernel_assign_ops([l], **params)
        logger.debug('Kernel ops for #{}: {}'.format(l, layer_kernel_ops))
        ops.append(layer_kernel_ops)

    return ops
