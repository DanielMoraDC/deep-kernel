import tensorflow as tf
import numpy as np

import logging
import collections

from layout import kernel_example_layout_fn
from ops import get_model_weights, loss_ops_list, get_accuracy_op, \
                train_ops_list, get_l2_ops_list, get_kernel_assign_ops_list

from protodata.data_ops import DataMode

logger = logging.getLogger(__name__)


RunContext = collections.namedtuple(
    'RunContext',
    [
        'logits_op', 'train_ops', 'loss_ops', 'acc_op', 'step_op',
        'steps_per_epoch', 'l2_ops', 'lr_op', 'summary_op',
        'kernel_assign_ops'
    ]
)


class RunStatus(object):

    def __init__(self, loss=None, acc=None, l2=None):
        self._loss = loss if loss is not None else []
        self._acc = acc if acc is not None else []
        self._l2 = l2 if l2 is not None else []
        self.epoch = None

    def update(self, loss, acc, l2):
        self._loss.append(loss)
        self._acc.append(acc)
        self._l2.append(l2)

    def clear(self):
        self._loss, self._acc, self._l2 = [], [], []

    def loss(self):
        return np.mean(self._loss)

    def acc(self):
        return np.mean(self._acc)

    def l2(self):
        return np.mean(self._l2)

    def error(self):
        return 1 - np.mean(self._acc)


def run_training_epoch_debug_weights(sess, context, layer_idx, num_layers):
    status = RunStatus()

    logger.debug('Running training epoch on {} variables'.format(layer_idx))

    weight_ops = []
    for i in range(1, num_layers + 1):
        if i == num_layers:
            include_output = True
        else:
            include_output = False

        current_weights = get_model_weights([i], include_output)

        for weight in current_weights:
            logger.info('Adding %s' % weight.name)
            weight_ops.append(weight)

    for i in range(context.steps_per_epoch):
        results = sess.run(
            [
                context.train_ops[layer_idx],
                context.loss_ops[layer_idx],
                context.acc_op,
                context.l2_ops[layer_idx],
            ] + weight_ops
        )

        _, loss, acc, l2 = results[:4]
        weights = results[4:]

        for i, w in enumerate(weights):
            trained = '[Trained]' if i + 1 == layer_idx or layer_idx == 0 \
                else ''
            logger.info(
                'First value layer %d %.10f %s\n'
                % (i + 1, w[0, 0], trained)
            )
        logger.info('Ended step\n')

        status.update(loss, acc, l2)

    return status


def run_training_epoch_debug_l2(sess, context, layer_idx, num_layers):
    status = RunStatus()

    logger.debug('Running training epoch on {} variables'.format(layer_idx))

    l2_ops = [context.l2_ops[i] for i in range(0, num_layers+1)]

    for i in range(context.steps_per_epoch):
        results = sess.run(
            [
                context.train_ops[layer_idx],
                context.loss_ops[layer_idx],
                context.acc_op,
                context.l2_ops[layer_idx],
            ] + l2_ops
        )

        _, loss, acc, l2_truth = results[:4]
        l2_results = results[4:]

        for i, l2 in enumerate(l2_results):
            trained = '[Trained]' if i == layer_idx or layer_idx == 0 \
                else ''
            num = str('layer %d + output' % i) if i != 0 else 'all'
            logger.info(
                'L2 {0:20} {1:.8f} {2}'.format(
                    num, l2, trained
                )
            )

        out_l2 = (np.sum(l2_results[1:]) - l2_results[0])/(num_layers-1)
        logger.info(
            'L2 {0:20} {1:.8f} [Trained]'.format(
                'output_layer', out_l2
            )
        )
        logger.info('Ended step\n')

        status.update(loss, acc, l2_truth)

    return status


def run_training_epoch(sess, context, layer_idx):
    status = RunStatus()

    logger.debug('Running training epoch on {} variables'.format(layer_idx))

    for i in range(context.steps_per_epoch):
        _, loss, acc, l2 = sess.run(
            [
                context.train_ops[layer_idx],
                context.loss_ops[layer_idx],
                context.acc_op,
                context.l2_ops[layer_idx],
            ]
        )

        if context.kernel_assign_ops is not None:
            sess.run(context.kernel_assign_ops[layer_idx])

        status.update(loss, acc, l2)

    # DEBUG
    '''
    kernel_op_w = tf.get_collection('KERNEL_VARS')[0]
    kernel_op_b = tf.get_collection('KERNEL_VARS')[1]

    _, loss, acc, l2, aux_w, aux_b = sess.run(
        [
            context.train_ops[layer_idx],
            context.loss_ops[layer_idx],
            context.acc_op,
            context.l2_ops[layer_idx],
            kernel_op_w,
            kernel_op_b
        ]
    )
    '''

    # DEBUG
    '''
    print('{}: {}'.format(kernel_op_w, aux_w))
    print('{}: {}'.format(kernel_op_b, aux_b))
    '''

    # Update epoch
    epoch = sess.run(context.step_op)
    status.epoch = epoch

    return status


def eval_epoch(sess, context, layer_idx):
    status = RunStatus()

    for _ in range(context.steps_per_epoch):
        loss, acc, l2 = sess.run(
            [
                context.loss_ops[layer_idx],
                context.acc_op,
                context.l2_ops[layer_idx]
            ]
        )

        status.update(loss, acc, l2)

    return status


def test_step(sess, test_context):
    return sess.run([
        test_context.loss_ops[0],
        test_context.acc_op,
        test_context.l2_ops[0]
    ])


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
    lr_decay_epochs = params.get('lr_decay_epochs', 500)
    network_fn = params.get('network_fn', kernel_example_layout_fn)
    batch_size = params.get('batch_size')
    memory_factor = params.get('memory_factor')
    n_threads = params.get('n_threads')
    n_layers = params.get('num_layers')

    if folds is not None:
        fold_size = dataset.get_fold_size()
        steps_per_epoch = int(fold_size * len(folds) / batch_size)
    else:
        steps_per_epoch = None

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

        loss_ops = loss_ops_list(
            logits=logits,
            y=labels,
            sum_collection=tag,
            n_classes=dataset.get_num_classes(),
            **params
        )

        if is_training:

            # Increase training epoch
            step_op = tf.assign(step, step + 1)

            # Exp decaying learning rate:
            #   lr(t)' = lr * decay^(step/decay_steps)
            # Equivalent to multiply by 'decay' every
            # 'decay_steps' epochs
            lr_op = tf.train.exponential_decay(
                lr,
                step,
                decay_steps=lr_decay_epochs,
                decay_rate=lr_decay,
                staircase=True
            )

            train_ops = train_ops_list(lr_op, loss_ops, n_layers, tag)
            kernel_assign_ops = get_kernel_assign_ops_list(**params) \
                if params.get('kernel_dropout_rate', None) is not None \
                else None
        else:
            train_ops, lr_op, step_op = None, None, None
            kernel_assign_ops = None

        # Evaluate model
        accuracy_op = get_accuracy_op(
            logits, labels, dataset.get_num_classes()
        )

        summary_op = tf.summary.merge_all(tag)

    return RunContext(
        logits_op=logits,
        train_ops=train_ops,
        loss_ops=loss_ops,
        lr_op=lr_op,
        acc_op=accuracy_op,
        steps_per_epoch=steps_per_epoch,
        summary_op=summary_op,
        l2_ops=get_l2_ops_list(**params),
        step_op=step_op,
        kernel_assign_ops=kernel_assign_ops
    )
