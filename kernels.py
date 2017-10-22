import numpy as np
import tensorflow as tf


class KernelFunction(object):

    """ Future abstract class. So far represents a Random Fourier kernel """

    def __init__(self, kernel_size=32, use_random=False):
        self._use_random = use_random
        self._matrix = None
        self._kernel_size = kernel_size

    def apply_kernel(self, x, dims, mean=0.0, std=1.0):
        if self._use_random:
            # Computes a random tensor each time
            random_tensor = tf.random_normal(
                shape=[dims, self._kernel_size], mean=mean, stddev=std
            )
        else:
            # Computes a random tensor once and keeps it
            if self._matrix is None:
                self._matrix = np.random.normal(
                    mean, std, [dims, self._kernel_size]
                )
            random_tensor = tf.constant(self._matrix, dtype=tf.float32)

        cos = tf.cos(tf.matmul(x, random_tensor))
        return tf.divide(cos, 1/np.sqrt(dims))
