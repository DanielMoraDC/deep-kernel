import tensorflow as tf

from kernels import KernelFunction


def example_layout_fn(x, outputs, **params):
    output_units = _map_classes_to_output(outputs)
    hidden_units = params.get('hidden_units', 128)
    inputs = _input_layer(x, **params)
    hidden = _fully_connected(inputs, hidden_units)
    return _fully_connected(hidden, output_units, activation_fn=None)


def kernel_example_layout_fn(x, outputs, **params):
    output_units = _map_classes_to_output(outputs)
    hidden_units = params.get('hidden_units', 128)
    kernel_size = params.get('kernel_size', 64)
    kernel_std = params.get('kernel_std', 32)

    inputs = _input_layer(x, **params)
    kernel = KernelFunction(kernel_size=kernel_size)
    # TODO: test whether we should use relu or no activation
    hidden = _fully_connected(inputs, hidden_units, activation_fn=None)
    hidden_kernel = kernel.apply_kernel(
        hidden, dims=hidden_units, std=kernel_std)
    return _fully_connected(hidden_kernel, output_units, activation_fn=None)


def _map_classes_to_output(outputs):
    if outputs == 2:
        # Logistic regression
        return 1
    elif outputs > 2:
        return outputs
    else:
        raise ValueError('Number of outputs must be at least 2')


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
