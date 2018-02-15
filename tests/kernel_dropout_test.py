import unittest
import numpy as np
import tensorflow as tf

from kernels import _generate_w_mask, kernel_dropout

PERSIST_PROB = 0.50

W = np.array([
    [0, 1, 2, 3, 4],
    [20, 15, 10, 5, 0],
    [21, 16, 11, 6, 1],
    [22, 17, 12, 7, 2],
    [23, 18, 14, 8, 3]
])

W_SAMPLE = np.array([
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1]
])


class KernelDropoutTestCase(unittest.TestCase):

    def _mask_test(self, x):
        pl = tf.placeholder(dtype=tf.float32, shape=(x.shape))

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            mask, mask_inv = sess.run(
                _generate_w_mask(pl, keep_ratio=PERSIST_PROB),
                feed_dict={pl: x}
            )

            # Make sure they do not overlap
            self.assertTrue(np.array_equal(np.ones(mask.shape), mask+mask_inv))
            self.assertFalse(True)

    def _dropout_test(self, x, x_sample):
        pl = tf.placeholder(dtype=tf.float32, shape=(x.shape))
        sample = tf.placeholder(dtype=tf.float32, shape=(x_sample.shape))

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            dropout = sess.run(
                kernel_dropout(pl, sample, keep_ratio=PERSIST_PROB),
                feed_dict={pl: x, sample: x_sample}
            )

            x_eq_rows_arr = np.nonzero(np.equal(dropout, x).all(axis=1))[0]
            sample_eq_rows_arr = np.nonzero(
                np.equal(dropout, x_sample).all(axis=1)
            )[0]

            x_rows_set = set(list(x_eq_rows_arr))
            sample_rows_set = set(list(sample_eq_rows_arr))

            # Both matrixes do not overlap
            self.assertTrue(
                x_rows_set.intersection(sample_rows_set) == set([])
            )

            # Every row in the dropout matrix is contained in inputs
            self.assertTrue(
                x_rows_set.union(sample_rows_set) == set(range(x.shape[0]))
            )

            self.assertFalse(True)

    def test_mask_gen_w(self):
        self._mask_test(W)

    def test_dropout_w(self):
        self._dropout_test(W, W_SAMPLE)


if __name__ == '__main__':
    unittest.main()
