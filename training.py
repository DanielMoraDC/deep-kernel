import tensorflow as tf
import numpy as np

import logging
import collections
import os

from kernels import KERNEL_ASSIGN_OPS
from layout import get_layer_id, kernel_example_layout_fn

from protodata.data_ops import DataMode

logger = logging.getLogger(__name__)


RunContext = collections.namedtuple(
    'RunContext',
    ['logits_op', 'train_ops', 'loss_op', 'acc_op', 'steps_per_epoch']
)


def run_training_epoch(sess, context, layer_idx):
    epoch_loss, epoch_accs = [], []

    logger.info('Running training epoch on {} variables'.format(layer_idx))

    for i in range(context.steps_per_epoch):
        _, loss, acc = sess.run(
            [context.train_ops[layer_idx], context.loss_op, context.acc_op],
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
    num = np.sum(strip)
    den = np.min(strip)*k
    return 1000 * ((num/den) - 1) if den != 0.0 else 0.0


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


def l1_norm(weights):
    return tf.add_n([tf.reduce_sum(tf.abs(x)) for x in weights])


def l2_norm(weights):
    return tf.add_n([tf.nn.l2_loss(x) for x in weights])


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


def save_model(monitored_sess, saver, folder, step):
    path = os.path.join(folder, 'model_' + str(step) + '.ckpt')
    sess = monitored_sess._sess._sess._sess._sess
    saver.save(sess, path)
    return path


def init_kernel_ops(sess):
    sess.run(tf.get_collection(KERNEL_ASSIGN_OPS))


def get_loss_op(logits, y, sum_collection, n_classes, **params):
    l1_ratio = params.get('l1_ratio', None)
    l2_ratio = params.get('l2_ratio', None)

    l1_term = l1_norm(get_model_weights()) * l1_ratio \
        if l1_ratio is not None else tf.constant(0.0)
    tf.summary.scalar('l1_term', l1_term, [sum_collection])

    l2_term = l2_norm(get_model_weights()) * l2_ratio \
        if l2_ratio is not None else tf.constant(0.0)
    tf.summary.scalar('l2_term', l2_term, [sum_collection])

    loss_term = tf.reduce_mean(
        get_loss_fn(
            logits, y, n_classes
        )
    )

    tf.summary.scalar('loss_term', loss_term, [sum_collection])

    loss_op = loss_term + l1_term + l2_term
    tf.summary.scalar('total_loss', loss_op, [sum_collection])

    return loss_op


def build_run_context(dataset,
                      reader,
                      tag,
                      folds,
                      step,
                      reuse=False,
                      is_training=True,
                      **params):
    lr = params.get('lr', 0.01)
    lr_decay = params.get('lr_decay', 0.5)
    lr_decay_epocs = params.get('lr_decay_epochs', 500)
    network_fn = params.get('network_fn', kernel_example_layout_fn)
    batch_size = params.get('batch_size')
    memory_factor = params.get('memory_factor')
    n_threads = params.get('n_threads')

    if folds is not None:
        fold_size = dataset.get_fold_size()
        steps_per_epoch = int(fold_size * len(folds) / batch_size)
        lr_decay_steps = lr_decay_epocs * steps_per_epoch
    else:
        steps_per_epoch = None
        lr_decay_steps = 10000  # Default value, not used

    data_subset = DataMode.TRAINING if tag == DataMode.VALIDATION else tag
    features, labels = reader.read_folded_batch(
        batch_size=batch_size,
        data_mode=data_subset,
        folds=folds,
        memory_factor=memory_factor,
        reader_threads=n_threads,
        train_mode=is_training,
        shuffle=True
    )

    scope_params = {'reuse': reuse}
    with tf.variable_scope("network", **scope_params):

        logits = network_fn(features,
                            columns=dataset.get_wide_columns(),
                            outputs=dataset.get_num_classes(),
                            tag=tag,
                            is_training=is_training,
                            **params)

        loss_op = get_loss_op(
            logits=logits,
            y=labels,
            sum_collection=tag,
            n_classes=dataset.get_num_classes(),
            **params
        )

        if is_training:

            n_layers = params.get('num_layers')

            # Decaying learning rate: lr(t)' = lr / (1 + decay * t)
            decayed_lr = tf.train.inverse_time_decay(
                lr, step, decay_steps=lr_decay_steps, decay_rate=lr_decay
            )
            tf.summary.scalar('lr', decayed_lr, [tag])

            train_ops = train_ops_list(step, decayed_lr, loss_op, n_layers)
        else:
            train_ops = None

        # Evaluate model
        accuracy_op = get_accuracy_op(
            logits, labels, dataset.get_num_classes()
        )
        tf.summary.scalar('accuracy', accuracy_op, [tag])

    return RunContext(
        logits_op=logits,
        train_ops=train_ops,
        loss_op=loss_op,
        acc_op=accuracy_op,
        steps_per_epoch=steps_per_epoch,
    )


def variables_from_layers(layer_list, include_output=True):
    selected = []
    train_vars = tf.trainable_variables()
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


def get_restore_info(num_layers, prev_layer_folder):
    vars_to_restore = variables_from_layers(
        range(1, num_layers), include_output=False
    )
    restore_saver = tf.train.Saver(var_list=vars_to_restore)
    return restore_saver, prev_layer_folder, vars_to_restore


def train_ops_list(step, lr, loss_op, n_layers):
    """
    Builds a tensor with training ops where the ith position
    corresponds to the operation to train layer i. The zero position
    is the function where we optimize everything
    """
    train_ops = []

    # First position is for all
    train_ops.append(
        get_train_op(step, lr, loss_op, tf.trainable_variables())
    )
    logger.info('Optimizer {} uses {}'.format(0, tf.trainable_variables()))

    for i in range(1, n_layers + 1):
        opt_vars = variables_from_layers(i, True)
        logger.info('Optimizer {} uses {}'.format(i, opt_vars))
        train_ops.append(
            get_train_op(step, lr, loss_op, opt_vars)
        )

    return train_ops


def get_train_op(step, lr, loss_op, opt_var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # This is needed for the batch norm moving averages
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(
            loss_op, var_list=opt_var_list, global_step=step
        )

    return train_op
