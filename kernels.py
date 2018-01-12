import abc

import numpy as np
import tensorflow as tf

KERNEL_COLLECTION = 'KERNEL_VARS'
KERNEL_ASSIGN_OPS = 'KERNEL_ASSIGN_OPS'


class KernelFunction(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, name, input_dims, kernel_size):
        self._name = name
        self._input_dims = input_dims
        self._kernel_size = kernel_size

    def apply_kernel(self, x, tag):
        """
        Applies a kernel function on the given vector
        """


class RandomFourierFeatures(KernelFunction):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def draw_samples(self, input_size, rff_features):
        """
        Draws samples according to the corresponding Fourier distribution
        """

    def apply_kernel(self, x, tag):

        # Create variable
        w = tf.get_variable(
            self._name,
            [self._input_dims, self._kernel_size],
            trainable=False,  # Important: this is constant!,
            collections=[KERNEL_COLLECTION, tf.GraphKeys.GLOBAL_VARIABLES]
        )

        w_value = self.draw_samples()

        assign_op = w.assign(w_value)
        tf.add_to_collection(KERNEL_ASSIGN_OPS, assign_op)

        # Let's store the matrix object so we have it if needed
        self._w = w

        # Check: see matrix does not change with time
        tf.summary.histogram(self._name + '_w', w, [tag])

        # Check: input to be centered around 0
        z = tf.matmul(x, w)
        tf.summary.histogram(self._name + '_z', z, [tag])

        # We assume input is centered around 0. Since cos of 0 is 1,
        # output would be shifted to the right limit of the axes.
        # Adding pi/2 we center the output of the cosinus around 0,
        # which is good if we have sigmoid or tanh activations
        z_centered = z + tf.constant(np.pi/2.0)
        tf.summary.histogram(
            self._name + '_z_centered', z_centered, [tag]
        )

        # Difference from orifinal paper: empirical results show that
        # by diving by a constant at each step we make the output of each
        # progressively decrease and therefore and we get much higher error
        cos = tf.cos(z_centered) + np.sqrt(1/self._kernel_size)
        return cos


class GaussianRFF(RandomFourierFeatures):

    def __init__(self, name, input_dims, kernel_size=32, mean=0.0, std=1.0):
        super(GaussianRFF, self).__init__(
            name, input_dims, kernel_size
        )
        self._mean = mean
        self._std = std

    def draw_samples(self):
        return np.random.normal(
            self._mean,
            self._std,
            [self._input_dims, self._kernel_size]
        )
