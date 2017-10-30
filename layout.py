import tensorflow as tf

from kernels import RandomFourierFeatures


def example_layout_fn(x, outputs, **params):
    output_units = _map_classes_to_output(outputs)
    hidden_units = params.get('hidden_units', 128)

    inputs = _input_layer(x, name='input', **params)

    hidden = _fully_connected(inputs, hidden_units, name='hidden')

    return _fully_connected(
        hidden, output_units, 'output', activation_fn=None
    )


def kernel_example_layout_fn(x, outputs, **params):
    output_units = _map_classes_to_output(outputs)
    hidden_units = params.get('hidden_units', 128)
    kernel_size = params.get('kernel_size', 64)
    kernel_std = params.get('kernel_std', 32)
    kernel = RandomFourierFeatures(
        name='kernel_layer',
        input_dims=hidden_units,
        std=kernel_std,
        kernel_size=kernel_size
    )

    inputs = _input_layer(x, name='input', **params)

    hidden = _fully_connected(
        inputs, hidden_units, name='hidden', activation_fn=None
    )

    hidden_kernel = kernel.apply_kernel(hidden)

    return _fully_connected(
        hidden_kernel, output_units, name='output', activation_fn=None
    )


def _map_classes_to_output(outputs):
    if outputs == 2:
        # Logistic regression
        return 1
    elif outputs > 2:
        return outputs
    else:
        raise ValueError('Number of outputs must be at least 2')


def _fully_connected(x, outputs, name, activation_fn=tf.nn.relu):
    return tf.contrib.layers.fully_connected(
        x,
        outputs,
        activation_fn=activation_fn,
        variables_collections=[tf.GraphKeys.WEIGHTS],
        scope=name
    )


def _input_layer(x, name, **params):
    return tf.contrib.layers.input_from_feature_columns(
        x,
        params['columns'],
        name
    )
