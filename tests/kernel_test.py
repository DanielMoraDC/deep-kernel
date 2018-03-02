import tensorflow as tf
import numpy as np

from kernels import GaussianRFF, KERNEL_ASSIGN_OPS

import unittest
import logging

logger = logging.getLogger(__name__)


class KernelTestCase(unittest.TestCase):

    def groundtruth_result(self, x, w, b):
        n, D = x.shape[0], w.shape[0]
        result = np.zeros((n, D))

        for i in range(n):
            for j in range(D):
                const = np.sqrt(2/D)
                z = const * np.cos(np.dot(x[i, ...], w[j, ...]) + b[j])
                result[i, j] = z

        return result

    def test_rff(self):
        n, inp_dim, kernel_size = 5, 3, 6
        kernel_mean = 0.0
        kernel_std = 0.1

        x = np.random.random((n, inp_dim))
        inputs = tf.placeholder(shape=[None, inp_dim], dtype=tf.float32)

        kernel = GaussianRFF(
            name='layer_kernel',
            input_dims=inp_dim,
            kernel_mean=kernel_mean,
            kernel_std=kernel_std,
            kernel_size=kernel_size,
        )

        kernel_op = kernel.apply_kernel(inputs, 'training')

        with tf.Session() as sess:

            # Assign matrix value
            sess.run(tf.get_collection(KERNEL_ASSIGN_OPS))

            kernel_out, kernel_mat, kernel_b = sess.run(
                [kernel_op, kernel._w, kernel._b],
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

            # Check the vectorized version is correct
            gt = self.groundtruth_result(x, kernel_mat, kernel_b)
            self.assertTrue(np.all(np.isclose(gt, kernel_out, rtol=0.01)))

            # Check size is correct
            self.assertTrue(kernel_out.shape == (n, kernel_size))

            # Check is around 0
            self.assertTrue(np.all(np.isclose(kernel_out, 0.0, atol=2.0)))


if __name__ == '__main__':
    unittest.main()
