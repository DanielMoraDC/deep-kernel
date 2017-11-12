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

    def apply_kernel(self, x, tag):
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

    def apply_kernel(self, x, tag):

        if _exists_variable(self._name):
            # Get existing variable
            matrix = tf.get_variable(self._name)
        else:
            # Create variable
            matrix = tf.get_variable(
                self._name,
                [self._input_dims, self._kernel_size],
                trainable=False  # Important: this is constant!
            )

            matrix_value = np.random.normal(
                self._mean,
                self._std,
                [self._input_dims, self._kernel_size]
            )

            assign_op = matrix.assign(matrix_value)
            tf.add_to_collection(KERNEL_ASSIGN_OPS, assign_op)

        # Check: see matrix does not change with time
        tf.summary.histogram(self._name + '_matrix', matrix, [tag])
        
        # Check: input to be centered around 0
        matrix_mul = tf.matmul(x, matrix)
        tf.summary.histogram(self._name + '_matrix_mul', matrix_mul, [tag])
        
        # Check: input to be centered around pi/2
        matrix_mul_centeterd = matrix_mul + tf.constant(np.pi/2.0) 
        tf.summary.histogram(
            self._name + '_matrix_mul_centered', matrix_mul_centeterd, [tag]
        )
        
        # We assume input is centered around 0. Since cos of 0 is 1,
        # output would be shifted to the right limit of the axes.
        # Adding pi/2 we center the output of the cosinus around 0
        cos = tf.cos(matrix_mul_centeterd)
        return tf.divide(cos, np.sqrt(self._kernel_size))


def _exists_variable(name):
    """
    Returns whether variable exists in the current scope
    """
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    for var in all_vars:
        if var == name:
            return True
    return False
