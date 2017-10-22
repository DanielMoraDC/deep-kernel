import tensorflow as tf
from training import get_accuracy_op

import unittest


class AccuracyTestCase(unittest.TestCase):

    def test_multilabel(self):
        logits = tf.placeholder(shape=[None, 4], dtype=tf.float32)
        labels = tf.placeholder(shape=[None, 1], dtype=tf.int32)

        with tf.Session() as sess:
            acc = sess.run(
                get_accuracy_op(logits, labels, 4),
                feed_dict={
                    logits: [
                        [0.1, 0.2, 0.3, 0.4],
                        [0.60, 0.1, 0.1, 0.2],
                        [0.1, 0.40, 0.45, 0.05],
                        [0.05, 0.35, 0.20, 0.40]
                    ],
                    labels: [[0], [1], [2], [3]]
                }
            )

        self.assertTrue(acc, 0.50)

    def test_two_classes(self):
        logits = tf.placeholder(shape=[5, 1], dtype=tf.float32)
        labels = tf.placeholder(shape=[5, 1], dtype=tf.int32)

        with tf.Session() as sess:
            acc = sess.run(
                get_accuracy_op(logits, labels, 2),
                feed_dict={
                    logits: [[-4], [-2], [0], [2], [4]],
                    labels: [[0], [0], [1], [1], [0]]
                }
            )

        self.assertTrue(acc, 0.80)


if __name__ == '__main__':
    unittest.main()
