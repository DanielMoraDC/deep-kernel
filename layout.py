import tensorflow as tf

from kernels import KernelFunction


def example_layout_fn(x, outputs, **params):
    inputs = _input_layer(x, **params)
    hidden = _fully_connected(inputs, 256)
    return _fully_connected(hidden, outputs)


def kernel_example_layout_fn(x, outputs, **params):
    inputs = _input_layer(x, **params)
    kernel = KernelFunction()
    hidden = _fully_connected(inputs, 256)
    hidden_kernel = kernel.apply_kernel(hidden, dims=256)
    return _fully_connected(hidden_kernel, outputs, activation_fn=None)


def _fully_connected(x, outputs, activation_fn=tf.nn.relu):
    return tf.contrib.layers.fully_connected(
        x,
        outputs,
        activation_fn=activation_fn,
        variables_collections=[tf.GraphKeys.WEIGHTS]
    )


def _input_layer(x, **params):
    print(x)
    print(type(x))
    print(type(params['columns']))
    print(params['columns'])
    return tf.contrib.layers.input_from_feature_columns(
        x,
        params['columns'],
    )
