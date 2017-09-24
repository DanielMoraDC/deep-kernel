import numpy as np
import tensorflow as tf


class KernelFunction(object):

    """ Future abstract class. So far represents a Random Fourier kernel """

    def __init__(self, use_random=False):
        self._use_random = use_random
        self._random = None

    def apply_kernel(self, x, dims, mean=0.0, std=0.1):
        if self._random is None:
            self._random = tf.random_normal(shape=[dims],
                                            mean=mean,
                                            stddev=std)

        cos = tf.cos(tf.multiply(self._random, x))
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
