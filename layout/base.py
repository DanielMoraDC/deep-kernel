import tensorflow as tf
import logging

from kernels import GaussianRFF

logger = logging.getLogger(__name__)

INPUT_LAYER = 'input'
OUTPUT_LAYER = 'output'
BATCH_NORM_COLLECTION = 'BATCH_NORM'
LAYER_NAME = '{layer_id}_{layer_type}'


def get_layer_id(name):
    return float(name.split('_')[0])


def _fully_connected(x,
                     outputs,
                     idx,
                     tag,
                     is_training,
                     activation_fn=tf.nn.relu):

    name = LAYER_NAME.format(layer_id=idx, layer_type='fc')
    fc_layer = tf.contrib.layers.fully_connected(
        x,
        outputs,
        activation_fn=activation_fn,
        weights_initializer=tf.variance_scaling_initializer,
        variables_collections=[tf.GraphKeys.WEIGHTS],
        scope=name
    )
    tf.summary.histogram(name, fc_layer, [tag])
    return fc_layer


def fc_block(x, idx, tag, is_training, batch_norm=False, **params):
    hidden_units = params.get('hidden_units')
    hidden = _fully_connected(
        x=x,
        outputs=hidden_units,
        idx=idx,
        tag=tag,
        is_training=is_training,
        activation_fn=None
    )

    if batch_norm:
        # Update ops for moving average are automatically
        # placed in tf.GraphKeys.UPDATE_OPS
        hidden = tf.contrib.layers.batch_norm(
            hidden,
            center=True,
            scale=True,
            is_training=is_training,
            variables_collections=[BATCH_NORM_COLLECTION],
            scope=LAYER_NAME.format(layer_id=idx, layer_type='bn')
        )

    activated = params.get('activation_fn', tf.nn.relu)(hidden)

    if params.get('fc_dropout_keep_prob', None) is not None:
        logger.debug(
            "Enabled dropout (keep rate(%f)) on fc %s" %
            (params.get('fc_dropout_keep_prob'), idx)
        )
        activated = tf.contrib.layers.dropout(
            hidden,
            keep_prob=params.get('fc_dropout_keep_prob'),
            is_training=is_training
        )

    return activated


def _map_classes_to_output(outputs):
    if outputs == 2:
        # Logistic regression
        return 1
    elif outputs > 2:
        return outputs
    else:
        raise ValueError('Number of outputs must be at least 2')


def kernel_block(x, idx, tag, is_training, batch_norm=False, **params):

    hidden_units = params.get('hidden_units')
    kernel_size = params.get('kernel_size')
    kernel_std = params.get('kernel_std')

    hidden = _fully_connected(
        x=x,
        outputs=hidden_units,
        idx=idx,
        tag=tag,
        is_training=is_training,
        activation_fn=None
    )

    if batch_norm:
        # Update ops for moving average are automatically
        # placed in tf.GraphKeys.UPDATE_OPS
        hidden = tf.contrib.layers.batch_norm(
            hidden,
            center=True,
            scale=True,
            is_training=is_training,
            variables_collections=[BATCH_NORM_COLLECTION],
            scope=LAYER_NAME.format(layer_id=idx, layer_type='bn')
        )

    kernel = GaussianRFF(
        name=LAYER_NAME.format(layer_id=idx, layer_type='kernel'),
        input_dims=hidden_units,
        kernel_std=kernel_std,
        kernel_size=kernel_size,
    )
    mapped = kernel.apply_kernel(hidden, tag)

    if params.get('fc_dropout_keep_prob', None) is not None:
        logger.debug(
            "Enabled dropout (keep rate(%f)) on kernel_fc %s" %
            (params.get('fc_dropout_keep_prob'), idx)
        )
        mapped = tf.contrib.layers.dropout(
            hidden,
            keep_prob=params.get('fc_dropout_keep_prob'),
            is_training=is_training
        )

    return mapped
