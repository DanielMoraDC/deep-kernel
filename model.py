from sklearn.base import RegressorMixin
import tensorflow as tf

from training import create_global_step, get_model_weights, l1_norm, l2_norm
from layout import example_layout_fn, kernel_example_layout_fn

from protodata.data_ops import DataMode
from protodata.reading_ops import DataReader
from protodata.datasets.australian import AusSettings
from protodata.utils import get_data_location
from protodata.datasets import Datasets

# TODO: generate data if it does not exist


class DeepKernelModel(RegressorMixin):

    """ Class that combines fully connected and kernel functions """

    def fit(self, X=None, y=None, **params):

        data_settings_fn = params.get('data_settings_fn')
        data_location = params.get('data_location')

        folder = params.get('folder', 'training')
        summaries_steps = params.get('summaries_steps', 10)
        network_fn = params.get('network_fn', kernel_example_layout_fn)
        batch_size = params.get('batch_size', 64)

        with tf.Graph().as_default():

            step = create_global_step()  # noqa

            dataset = data_settings_fn(dataset_location=data_location)

            # Read batches from dataset
            reader = DataReader(dataset)
            features, labels = reader.read_batch(
                batch_size=batch_size,
                data_mode=DataMode.TRAINING,
                memory_factor=1,  # TODO: as params
                reader_threads=2,  # TODO: as params
                train_mode=True
            )

            logits = network_fn(features,
                                columns=dataset.get_wide_columns(),
                                outputs=dataset.get_num_classes())
            prediction = tf.nn.softmax(logits)

            loss_op = self.get_loss_op(logits=logits, y=labels, **params)

            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            train_op = optimizer.minimize(loss_op)

            # Evaluate model
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
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

                # Define coordinator to handle all threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                while not sess.should_stop():

                    for i in range(50000):

                        sess.run([train_op])

                        if i % 100 == 0:
                            loss, acc = sess.run([loss_op, accuracy])
                            print('[%d] Loss: %f, Accuracy: %f'
                                  % (i, loss, acc))

                    coord.request_stop()
                    coord.join(threads)

                    break

    # TODO: static method, could we moved outside
    def get_loss_op(self, logits, y, **params):
        num_classes = logits.get_shape().as_list()[-1]
        l1_ratio = params.get('l1_ratio', None)
        l2_ratio = params.get('l2_ratio', None)

        l1_term = l1_norm(get_model_weights()) * l1_ratio \
            if l1_ratio is not None else tf.constant(0.0)
        tf.summary.scalar('l1_term', l1_term)

        l2_term = l2_norm(get_model_weights()) * l2_ratio \
            if l2_ratio is not None else tf.constant(0.0)
        tf.summary.scalar('l2_term', l2_term)

        loss_term = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(y, depth=num_classes))
        )

        tf.summary.scalar('loss_term', loss_term)

        loss_op = loss_term + l1_term + l2_term
        tf.summary.scalar('total_loss', loss_op)

        return loss_op

    def predict(self, X, y, **params):
        return None


if __name__ == '__main__':

    a = DeepKernelModel()
    a.fit(data_settings_fn=AusSettings,
          folder='/media/walle/815d08cd-6bee-4a13-b6fd-87ebc1de2bb0/walle/kernel',
          l2_ratio=0.01,
          data_location=get_data_location(Datasets.AUS))
