import tensorflow as tf
import logging

from kernels import GaussianRFF
from layout.base import _fully_connected, fc_block, LAYER_NAME, \
                   _map_classes_to_output, kernel_block, \
                   BATCH_NORM_COLLECTION

logger = logging.getLogger(__name__)


def cnn_example_layout_fn(x,
                          dataset,
                          tag,
                          is_training,
                          num_layers,
                          **params):

    # By default we will use 2 FC layers
    fc_layers = params.get('fc_layers', 2)

    logger.debug(
        'Building CNN network with %s data(training=%s)'
        % (tag, str(is_training)) +
        ' with %d cnn layers and %d fc layers' %
        (num_layers, fc_layers)
    )

    inputs = x['image']
    inputs = inputs / 255.0
    tf.summary.image("input", inputs, 3, [tag])

    block_out = inputs
    for i in range(1, num_layers+1):
        block_out = cnn_block(block_out, i, is_training, **params)

    block_out = tf.layers.flatten(block_out)

    for i in range(1, fc_layers+1):
        # We will treat FC blocks in a specific way due to its naming
        block_out = fc_block(
            block_out, 'fc_%d' % i, tag, is_training, **params
        )

    return _fully_connected(
        block_out,
        _map_classes_to_output(dataset.get_num_classes()),
        'output',
        tag,
        is_training,
        activation_fn=None
    )


def cnn_kernel_example_layout_fn(x,
                                 dataset,
                                 tag,
                                 is_training,
                                 num_layers,
                                 **params):

    # By default we will use 2 FC layers
    fc_layers = params.get('fc_layers', 2)

    logger.debug(
        'Building kernel CNN network with %s data(training=%s)'
        % (tag, str(is_training)) +
        ' with %d cnn layers and %d fc layers' %
        (num_layers, fc_layers)
    )

    inputs = x['image']
    inputs = inputs / 255.0
    tf.summary.image("input", inputs, 3, [tag])

    block_out = inputs
    for i in range(1, num_layers+1):
        block_out = cnn_kernel_block(block_out, i, tag, is_training, **params)

    block_out = tf.layers.flatten(block_out)

    for i in range(1, fc_layers+1):
        # We will treat FC blocks in a specific way due to its naming
        block_out = kernel_block(
            block_out, 'fc_%d' % i, tag, is_training, **params
        )

    return _fully_connected(
        block_out,
        _map_classes_to_output(dataset.get_num_classes()),
        'output',
        tag,
        is_training,
        activation_fn=None
    )


def cnn_block(x, idx, is_training, **params):

    out_conv = tf.contrib.layers.conv2d(
        x,
        activation_fn=None,
        num_outputs=params.get('map_size'),
        kernel_size=params.get('cnn_filter_size'),
        variables_collections=[tf.GraphKeys.WEIGHTS],
        stride=params.get('stride', 2),
        padding=params.get('padding', 'VALID'),
        scope=LAYER_NAME.format(layer_type='cnn', layer_id=str(idx))
    )

    tf.summary.histogram(
        '_'.join(['conv', str(idx)]), out_conv
    )

    if params.get('cnn_batch_norm', False):

        # Update ops for moving average are automatically
        # placed in tf.GraphKeys.UPDATE_OPS
        out_conv = tf.contrib.layers.batch_norm(
            out_conv,
            center=True,
            scale=True,
            is_training=is_training,
            variables_collections=[BATCH_NORM_COLLECTION],
            scope=LAYER_NAME.format(
                layer_type='conv_batch_norm', layer_id=str(idx)
            )
        )

        tf.summary.histogram(
            '_'.join(['batched_conv_', str(idx)]), out_conv
        )

    activation_fn = params.get('activation_fn', tf.nn.relu)
    activated = activation_fn(out_conv)

    tf.summary.histogram(
        '_'.join(['activated_conv', str(idx)]), out_conv
    )

    return activated


def cnn_kernel_block(x, idx, tag, is_training, **params):

    cnn_filter_size = params.get('cnn_filter_size')
    cnn_kernel_size = params.get('cnn_kernel_size')
    map_size = params.get('map_size')
    kernel_std = params.get('kernel_std')

    hidden = tf.contrib.layers.conv2d(
        x,
        activation_fn=None,
        num_outputs=map_size,
        kernel_size=cnn_filter_size,
        stride=params.get('stride', 2),
        padding=params.get('padding', 'VALID'),
        variables_collections=[tf.GraphKeys.WEIGHTS],
        biases_initializer=False,
        scope=LAYER_NAME.format(layer_type='cnn', layer_id=str(idx))
    )

    kernel = GaussianRFF(
        name=LAYER_NAME.format(layer_id=idx, layer_type='cnn_kernel'),
        input_dims=map_size,
        kernel_std=kernel_std,
        kernel_size=cnn_kernel_size,
    )

    if params.get('cnn_batch_norm', False):

        # Update ops for moving average are automatically
        # placed in tf.GraphKeys.UPDATE_OPS
        hidden = tf.contrib.layers.batch_norm(
            hidden,
            center=True,
            scale=True,
            is_training=is_training,
            variables_collections=[BATCH_NORM_COLLECTION],
            scope=LAYER_NAME.format(
                layer_type='conv_batch_norm', layer_id=str(idx)
            )
        )

    hidden = kernel.apply_kernel(hidden, tag)

    tf.summary.histogram(
        '_'.join(['kernel_out_cnn_', str(idx)]), hidden
    )

    return hidden
