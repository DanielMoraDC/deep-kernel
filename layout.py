import tensorflow as tf
import logging

from kernels import GaussianRFF


logger = logging.getLogger(__name__)

BATCH_NORM_COLLECTION = 'BATCH_NORM'
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


def fc_block(x, idx, tag, is_training, batch_norm=True, **params):
    hidden_units = params.get('hidden_units', 128)
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

    return tf.nn.relu(hidden)


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
    feature_input_size = x.get_shape().as_list()[1]

    kernel = GaussianRFF(
        name=LAYER_NAME.format(layer_id=idx, layer_type='kernel'),
        input_dims=feature_input_size,
        std=kernel_std,
        kernel_size=kernel_size,
    )

    transformed = kernel.apply_kernel(x, tag)

    return _fully_connected(
        x=transformed,
        outputs=hidden_units,
        idx=idx,
        tag=tag,
        is_training=is_training,
        activation_fn=None,
        use_bias=False
    )


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
                     activation_fn=tf.nn.relu,
                     use_bias=True):

    name = LAYER_NAME.format(layer_id=idx, layer_type='fc')
    bias_init = tf.zeros_initializer() if use_bias else None
    fc_layer = tf.contrib.layers.fully_connected(
        x,
        outputs,
        activation_fn=activation_fn,
        weights_initializer=tf.variance_scaling_initializer,
        variables_collections=[tf.GraphKeys.WEIGHTS],
        biases_initializer=bias_init,
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
