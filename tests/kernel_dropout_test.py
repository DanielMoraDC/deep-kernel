import unittest
import numpy as np
import tensorflow as tf

from kernels import kernel_dropout_w, kernel_dropout_b

PERSIST_PROB = 0.50

B = np.array([0, 1, 2, 3, 4])

B_SAMPLE = np.asarray([-1, -1, -1, -1, -1])

W = np.array([
    [0, 1, 2, 3, 4],
    [20, 15, 10, 5, 0],
    [21, 16, 11, 6, 1],
    [22, 17, 12, 7, 2],
])

W_SAMPLE = np.array([
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
])


class KernelDropoutTestCase(unittest.TestCase):

    def test_dropout_w(self):

        self.pl = tf.placeholder(dtype=tf.float32, shape=(W.shape))
        self.sample = tf.placeholder(dtype=tf.float32, shape=(W_SAMPLE.shape))

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            dropout = sess.run(
                kernel_dropout_w(
                    self.pl, self.sample, keep_ratio=PERSIST_PROB
                ),
                feed_dict={self.pl: W, self.sample: W_SAMPLE}
            )

            x_eq_rows_arr = np.nonzero(
                np.equal(dropout, W).all(axis=1)
            )[0]
            sample_eq_rows_arr = np.nonzero(
                np.equal(dropout, W_SAMPLE).all(axis=1)
            )[0]

            x_rows_set = set(list(x_eq_rows_arr))
            sample_rows_set = set(list(sample_eq_rows_arr))

            # Both matrices do not overlap
            self.assertTrue(
                x_rows_set.intersection(sample_rows_set) == set([])
            )

            # Every row in the dropout matrix is contained in inputs
            self.assertTrue(
                x_rows_set.union(sample_rows_set) == set(range(W.shape[0]))
            )

    def test_dropout_b(self):

        self.pl = tf.placeholder(dtype=tf.float32, shape=(B.shape))
        self.sample = tf.placeholder(dtype=tf.float32, shape=(B_SAMPLE.shape))

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            dropout = sess.run(
                kernel_dropout_b(
                    self.pl, self.sample, keep_ratio=PERSIST_PROB
                ),
                feed_dict={self.pl: B, self.sample: B_SAMPLE}
            )

            x_eq_rows_arr = np.nonzero(np.equal(dropout, B))[0]
            sample_eq_rows_arr = np.nonzero(np.equal(dropout, B_SAMPLE))[0]

            x_rows_set = set(list(x_eq_rows_arr))
            sample_rows_set = set(list(sample_eq_rows_arr))
            
            # Sample and original do not overlap
            self.assertTrue(
                x_rows_set.intersection(sample_rows_set) == set([])
            )

            # Every row in the dropout matrix is contained in inputs
            self.assertTrue(
                x_rows_set.union(sample_rows_set) == set(range(B.shape[0]))
            )


if __name__ == '__main__':
    unittest.main()

