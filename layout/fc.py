import tensorflow as tf
import logging

from layout.base import _fully_connected, _map_classes_to_output, \
                          fc_block, INPUT_LAYER, LAYER_NAME, kernel_block

logger = logging.getLogger(__name__)


def example_layout_fn(x, dataset, tag, is_training, num_layers=1, **params):

    logger.debug(
        'Building FC network with %s data(training=%s) with %d layers'
        % (tag, str(is_training), num_layers)
    )

    inputs = _input_layer(x, dataset, name=INPUT_LAYER)
    tf.summary.histogram("input", inputs, [tag])

    x = inputs
    for i in range(1, num_layers+1):
        x = fc_block(x, str(i), tag, is_training, **params)

    return _fully_connected(
        x,
        _map_classes_to_output(dataset.get_num_classes()),
        'output',
        tag,
        activation_fn=None
    )


def kernel_example_layout_fn(x,
                             dataset,
                             tag,
                             is_training,
                             num_layers=1,
                             **params):

    logger.debug(
        'Building kernel network with %s data(training=%s) with %d layers'
        % (tag, str(is_training), num_layers)
    )

    inputs = _input_layer(x, dataset, name=INPUT_LAYER)
    tf.summary.histogram("input", inputs, [tag])

    x = inputs
    for i in range(1, num_layers+1):
        layer_name = LAYER_NAME.format(
            net_id='kernel', layer_id=str(i), layer_type='nk'
        )
        x = kernel_block(x, layer_name, tag, is_training, **params)

    return _fully_connected(
        x,
        _map_classes_to_output(dataset.get_num_classes()),
        'output',
        tag,
        activation_fn=None
    )


def _input_layer(x, dataset, name):
    return tf.contrib.layers.input_from_feature_columns(
        x,
        dataset.get_wide_columns(),
        name
    )
