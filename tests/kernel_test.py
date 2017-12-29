import tensorflow as tf
import numpy as np

from kernels import RandomFourierFeatures, KERNEL_ASSIGN_OPS

import unittest
import logging

logger = logging.getLogger(__name__)


class KernelTestCase(unittest.TestCase):

    def test_multilabel(self):
        n, inp_dim, kernel_size = 5, 3, 6
        kernel_std = 0.1

        x = np.random.random((n, inp_dim))
        inputs = tf.placeholder(shape=[None, inp_dim], dtype=tf.float32)

        kernel = RandomFourierFeatures(
            name='layer_kernel',
            input_dims=inp_dim,
            std=kernel_std,
            kernel_size=kernel_size,
        )

        kernel_op = kernel.apply_kernel(inputs, 'training')

        with tf.Session() as sess:

            # Assign matrix value
            sess.run(tf.get_collection(KERNEL_ASSIGN_OPS))

            kernel_out, kernel_mat = sess.run(
                [kernel_op, kernel._matrix],
                feed_dict={inputs: x}
            )

            logger.info(
                'Input ({}): {}'.format(x.shape, x)
            )
            logger.info(
                'Output ({}): {}'.format(kernel_out.shape, kernel_out)
            )
            logger.info(
                'Kernel matrix ({}): {}'.format(kernel_mat.shape, kernel_mat)
            )

            # Check size is correct
            self.assertTrue(kernel_out.shape == (n, kernel_size))

            # Check is around 0
            self.assertTrue(np.all(np.isclose(kernel_out, 0.0, atol=2.0)))


if __name__ == '__main__':
    unittest.main()
