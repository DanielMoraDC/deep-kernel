import unittest
import tensorflow as tf
import numpy as np

from training import l2_norm

N_ROWS = 10
N_COLS = 5


def _paper_version(x):
    return tf.trace(tf.matmul(x, tf.transpose(x)))/2.0


def _tf_version(x):
    return l2_norm(x)


class AccuracyTestCase(unittest.TestCase):

    def setUp(self):
        self._matrix = np.random.random((N_ROWS, N_COLS))

    def test_l2equal(self):
        weight = tf.placeholder(shape=[N_ROWS, N_COLS], dtype=tf.float32)

        our_l2_tensor = _tf_version([weight])
        paper_tensor = _paper_version(weight)

        with tf.Session() as sess:
            ours, theirs = sess.run(
                [our_l2_tensor, paper_tensor],
                feed_dict={weight: self._matrix}
            )

        self.assertAlmostEqual(ours, theirs)


if __name__ == '__main__':
    unittest.main()
