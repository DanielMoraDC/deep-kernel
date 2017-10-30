from abc import ABCMeta

import numpy as np
import tensorflow as tf


KERNEL_ASSIGN_OPS = 'KERNEL_ASSIGN_OPS'


class KernelFunction(object):

    __metaclass__ = ABCMeta

    def __init__(self, name, input_dims, kernel_size):
        self._name = name
        self._input_dims = input_dims
        self._kernel_size = kernel_size

    def apply_kernel(self, x):
        """
        Applies a kernel function on the given vector
        """


class RandomFourierFeatures(KernelFunction):

    def __init__(self, name, input_dims, mean=0.0, std=1.0, kernel_size=32):
        super(RandomFourierFeatures, self).__init__(
            name, input_dims, kernel_size
        )
        self._mean = mean
        self._std = std

    def apply_kernel(self, x):

        if _exists_variable(self._name):
            # Get existing variable
            matrix = tf.get_variable(self._name)
        else:
            # Create variable
            matrix = tf.get_variable(
                self._name, [self._input_dims, self._kernel_size]
            )

            matrix_value = np.random.normal(
                self._mean,
                self._std,
                [self._input_dims, self._kernel_size]
            )

            assign_op = matrix.assign(matrix_value)
            tf.add_to_collection(KERNEL_ASSIGN_OPS, assign_op)

        cos = tf.cos(tf.matmul(x, matrix))
        return tf.divide(cos, 1/np.sqrt(self._input_dims))


def _exists_variable(name):
    """
    Returns whether variable exists in the current scope
    """
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    for var in all_vars:
        if var == name:
            return True
    return False
