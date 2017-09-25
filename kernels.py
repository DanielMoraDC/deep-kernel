import numpy as np
import tensorflow as tf


class KernelFunction(object):

    """ Future abstract class. So far represents a Random Fourier kernel """

    def __init__(self, use_random=True):
        self._use_random = use_random
        self._matrix = None

    def apply_kernel(self, x, dims, mean=0.0, std=1.0):
        if self._use_random:
            # Computes a random tensor each time
            random_tensor = tf.random_normal(
                shape=[dims], mean=mean, stddev=std
            )
        else:
            # Computes a random tensor once and keeps it
            if self._matrix is None:
                self._matrix = np.random.normal(mean, std, dims)
            random_tensor = tf.constant(self._matrix, dtype=tf.float32)

        cos = tf.cos(tf.multiply(random_tensor, x))
        return tf.divide(cos, 1/np.sqrt(dims))


def replicate_matrix(mat, times):
    """ Copies input matrix as many times as given by creating a batch """
    batched = tf.expand_dims(mat, 0)
    repetitions = tf.stack(
        [times] + [1] * len(mat.get_shape().as_list())
    )
    return tf.tile(batched, repetitions)


if __name__ == '__main__':

    hidden = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    batch_size = 2
    random = tf.constant([1.0, 2.0, 0.0]) 

    with tf.Session() as sess:

        y = tf.cos(tf.multiply(random, hidden))
        y = tf.divide(y, 1/np.sqrt(3))
        result = sess.run([y])
        print(result)
