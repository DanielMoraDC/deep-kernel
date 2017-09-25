from sklearn.base import RegressorMixin
import tensorflow as tf

from training import create_global_step, get_model_weights, l1_norm, l2_norm
from layout import example_layout_fn, kernel_example_layout_fn

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


class DeepKernelModel(RegressorMixin):

    """ Class that combines fully connected and kernel functions """

    def fit(self, X=None, y=None, **params):

        folder = params.get('folder', 'training')
        summaries_steps = params.get('summaries_steps', 10)
        network_fn = params.get('network_fn', kernel_example_layout_fn)
        num_classes = params.get('num_classes', 10)
        batch_size = params.get('batch_size', 32)

        with tf.Graph().as_default() as graph:

            step = create_global_step()

            x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
            y = tf.placeholder(shape=[None, num_classes], dtype=tf.float32)

            logits = network_fn(x)
            prediction = tf.nn.softmax(logits)

            loss_op = self.get_loss_op(logits=logits, y=y, **params)

            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            train_op = optimizer.minimize(loss_op)

            # Evaluate model
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            summary_op = tf.summary.merge_all()
            summary_hook = tf.train.SummarySaverHook(
                save_steps=summaries_steps,
                output_dir=folder,
                summary_op=summary_op
            )

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=folder,
                    hooks=[summary_hook]) as sess:

                while not sess.should_stop():

                    for i in range(1000):

                        batch_x, batch_y = mnist.train.next_batch(batch_size)
                        sess.run([train_op], {x: batch_x, y: batch_y})

                        if i % 10 == 0:
                            loss, acc = sess.run([loss_op, accuracy], {x: batch_x, y: batch_y})
                            print('[%d] Loss: %f, Accuracy: %f' % (i, loss, acc))


                    break

    # TODO: static method, could we moved outside
    def get_loss_op(self, logits, y, **params):
        l1_ratio = params.get('l1_ratio', None)
        l2_ratio = params.get('l2_ratio', None)

        l1_term = l1_norm(get_model_weights()) * l1_ratio \
            if l1_ratio is not None else tf.constant(0.0)
        tf.summary.scalar('l1_term', l1_term)

        l2_term = l2_norm(get_model_weights()) * l2_ratio \
            if l2_ratio is not None else tf.constant(0.0)
        tf.summary.scalar('l2_term', l2_term)

        loss_term = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        )

        tf.summary.scalar('loss_term', loss_term)

        loss_op = loss_term + l1_term + l2_term
        tf.summary.scalar('total_loss', loss_op)

        return loss_op

    def predict(self, X, y, **params):
        return None


if __name__ == '__main__':

    a = DeepKernelModel()
    a.fit(folder='/media/walle/815d08cd-6bee-4a13-b6fd-87ebc1de2bb0/walle/kernel',
          l2_ratio=0.01)
