import tensorflow as tf
import logging

from kernels import RandomFourierFeatures


logger = logging.getLogger(__name__)

INPUT_LAYER = 'input'
OUTPUT_LAYER = 'output'
LAYER_NAME = '{layer_id}_{layer_type}'


def example_layout_fn(x, outputs, tag, is_training, num_layers=1, **params):

    logger.debug(
        'Building FC network with %s data(training=%s) with %d layers'
        % (tag, str(is_training), num_layers)
    )

    inputs = _input_layer(x, name=INPUT_LAYER, **params)
    tf.summary.histogram("input", inputs, [tag])

    x = inputs
    for i in range(1, num_layers+1):
        x = fc_block(x, str(i), tag, is_training, **params)

    return _fully_connected(
        x,
        _map_classes_to_output(outputs),
        OUTPUT_LAYER,
        tag,
        is_training,
        activation_fn=None
    )


def fc_block(x, idx, tag, is_training, **params):
    hidden_units = params.get('hidden_units', 128)
    return _fully_connected(
        x,
        hidden_units,
        idx,
        tag,
        is_training,
    )


def kernel_example_layout_fn(x,
                             outputs,
                             tag,
                             is_training,
                             num_layers=1,
                             **params):

    logger.debug(
        'Building kernel network with %s data(training=%s) with %d layers'
        % (tag, str(is_training), num_layers)
    )

    inputs = _input_layer(x, name=INPUT_LAYER, **params)
    tf.summary.histogram("input", inputs, [tag])

    x = inputs
    for i in range(1, num_layers+1):
        layer_name = LAYER_NAME.format(
            net_id='kernel', layer_id=str(i), layer_type='nk'
        )
        x = kernel_block(x, layer_name, tag, is_training, **params)

    return _fully_connected(
        x,
        _map_classes_to_output(outputs),
        OUTPUT_LAYER,
        tag,
        is_training=is_training,
        activation_fn=None
    )


def kernel_block(x, idx, tag, is_training, **params):

    hidden_units = params.get('hidden_units', 128)
    kernel_size = params.get('kernel_size', 64)
    kernel_std = params.get('kernel_std', 32)

    kernel = RandomFourierFeatures(
        name=LAYER_NAME.format(layer_id=idx, layer_type='kernel'),
        input_dims=hidden_units,
        std=kernel_std,
        kernel_size=kernel_size,
    )

    hidden = _fully_connected(
        x,
        hidden_units,
        idx,
        tag,
        is_training=is_training,
        activation_fn=None
    )

    hidden_kernel = kernel.apply_kernel(hidden, tag)
    tf.summary.histogram("kernel_layer_" + idx, hidden_kernel, [tag])
    return hidden_kernel


def _map_classes_to_output(outputs):
    if outputs == 2:
        # Logistic regression
        return 1
    elif outputs > 2:
        return outputs
    else:
        raise ValueError('Number of outputs must be at least 2')


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


def _input_layer(x, name, **params):
    return tf.contrib.layers.input_from_feature_columns(
        x,
        params['columns'],
        name
    )


def get_layer_id(name):
    return float(name.split('_')[0])
