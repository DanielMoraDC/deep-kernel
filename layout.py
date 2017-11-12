import tensorflow as tf

from kernels import RandomFourierFeatures


def example_layout_fn(x, outputs, tag, is_training, **params):
    output_units = _map_classes_to_output(outputs)
    hidden_units = params.get('hidden_units', 128)
    batch_norm = params.get('batch_norm', True)

    inputs = _input_layer(x, name='input', **params)
    tf.summary.histogram("base/input", inputs, [tag])

    hidden = _fully_connected(
        inputs,
        hidden_units,
        'hidden',
        tag,
        is_training,
        batch_norm=batch_norm
    )

    return _fully_connected(
        hidden,
        output_units,
        'output',
        tag,
        is_training,
        batch_norm=False,
        activation_fn=None
    )


def kernel_example_layout_fn(x, outputs, tag, is_training, **params):
    output_units = _map_classes_to_output(outputs)
    hidden_units = params.get('hidden_units', 128)
    kernel_size = params.get('kernel_size', 64)
    kernel_std = params.get('kernel_std', 32)
    batch_norm = params.get('batch_norm', True)

    kernel = RandomFourierFeatures(
        name='kernel_layer',
        input_dims=hidden_units,
        std=kernel_std,
        kernel_size=kernel_size,
    )

    inputs = _input_layer(x, name='input', **params)
    tf.summary.histogram("kernel/input", inputs, [tag])

    hidden = _fully_connected(
        inputs,
        hidden_units,
        'hidden',
        tag,
        batch_norm=batch_norm,
        is_training=is_training,
        activation_fn=None
    )

    hidden_kernel = kernel.apply_kernel(hidden, tag)
    tf.summary.histogram("kernel/kernel_layer", hidden_kernel, [tag])

    if batch_norm:
        # Batch norm creates 2 moving averages (non-trainable)
        # And a trainable parameter beta
        # The update operations need to be controlled in the optimization:
        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm  # noqa
        hidden_kernel = tf.contrib.layers.batch_norm(
            hidden_kernel, is_training=is_training,
        )
        tf.summary.histogram(
            "kernel/kernel_layer_normed", hidden_kernel, [tag]
        )

    return _fully_connected(
        hidden_kernel,
        output_units,
        'output',
        tag,
        is_training=is_training,
        batch_norm=False,
        activation_fn=None
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
                     name,
                     tag,
                     is_training,
                     batch_norm,
                     activation_fn=tf.nn.relu):
    if batch_norm is True:
        fc_identity = tf.contrib.layers.fully_connected(
            x,
            outputs,
            activation_fn=activation_fn,
            variables_collections=[tf.GraphKeys.WEIGHTS],
            scope=name
        )
        tf.summary.histogram(name, fc_identity, [tag])

        batch_norm = tf.contrib.layers.batch_norm(
            fc_identity, is_training=is_training
        )
        tf.summary.histogram(name + '_batch_norm', batch_norm, [tag])

        layer_fn = tf.identity if activation_fn is None else activation_fn
        return layer_fn(batch_norm)
    else:
        fc_layer = tf.contrib.layers.fully_connected(
            x,
            outputs,
            activation_fn=activation_fn,
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
