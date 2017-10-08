import tensorflow as tf

from kernels import KernelFunction


def example_layout_fn(x, outputs, **params):
    hidden_units = params.get('hidden_units', 128)
    inputs = _input_layer(x, **params)
    hidden = _fully_connected(inputs, hidden_units)
    return _fully_connected(hidden, outputs)


def kernel_example_layout_fn(x, outputs, **params):
    hidden_units = params.get('hidden_units', 128)
    inputs = _input_layer(x, **params)
    kernel = KernelFunction()
    # TODO: test whether we should use relu or no activation
    hidden = _fully_connected(inputs, hidden_units, activation_fn=None)
    hidden_kernel = kernel.apply_kernel(hidden, dims=hidden_units)
    return _fully_connected(hidden_kernel, outputs, activation_fn=None)


def _fully_connected(x, outputs, activation_fn=tf.nn.relu):
    return tf.contrib.layers.fully_connected(
        x,
        outputs,
        activation_fn=activation_fn,
        variables_collections=[tf.GraphKeys.WEIGHTS]
    )


def _input_layer(x, **params):
    return tf.contrib.layers.input_from_feature_columns(
        x,
        params['columns'],
    )
