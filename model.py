from sklearn.base import RegressorMixin
import tensorflow as tf
import os

from training import create_global_step, get_model_weights, l1_norm, \
    l2_norm, get_accuracy, get_global_step, save_model
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
        summaries_steps = params.get('summaries_steps', 100)
        validation_interval = params.get('validation_interval', 100)
        max_steps = params.get('max_steps', 50000)
        train_tolerance = params.get('train_tolerance', 1e-3)

        # Tracking initialization
        prev_train_loss = float('inf')
        best_validation = {
            'loss': float('inf'), 'acc': None, 'step': 0
        }

        with tf.Graph().as_default() as graph:

            step = create_global_step()  # noqa

            # TODO this could be moved inside build_workflow
            dataset = data_settings_fn(dataset_location=data_location)
            reader = DataReader(dataset)

            # Get training operations
            train_logits, train_loss, train_op, train_acc = build_workflow(
                dataset, reader, DataMode.TRAINING, step, **params
            )

            # Get validation operations
            val_logits, val_loss, _, val_acc = build_workflow(
                dataset, reader, DataMode.VALIDATION, step, True, **params
            )

            # Initialize writers
            train_writer_path = os.path.join(folder, DataMode.TRAINING)
            os.makedirs(train_writer_path)
            train_writer = tf.summary.FileWriter(train_writer_path, graph)

            val_writer_path = os.path.join(folder, DataMode.VALIDATION)
            os.makedirs(val_writer_path)
            val_writer = tf.summary.FileWriter(val_writer_path, graph)

            # Gather summaries by dataset
            train_summary_op = tf.summary.merge_all(DataMode.TRAINING)
            val_summary_op = tf.summary.merge_all(DataMode.VALIDATION)

            saver = tf.train.Saver()

            with tf.train.MonitoredTrainingSession(
                    save_checkpoint_secs=None,
                    save_summaries_steps=None,
                    save_summaries_secs=None) as sess:

                # Define coordinator to handle all threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                while not sess.should_stop():

                    for i in range(max_steps):

                        sess.run([train_op])

                        if i % summaries_steps == 0:
                            sum_str, loss, acc, step_value = sess.run(
                                [train_summary_op, train_loss, train_acc, step]
                            )
                            train_writer.add_summary(sum_str, step_value)
                            print('[%d] Loss: %f, Accuracy: %f'
                                  % (step_value, loss, acc))

                            # Stop if training hasn't improved much
                            if loss > prev_train_loss or \
                                    (prev_train_loss - loss)/prev_train_loss < train_tolerance:
                                print('Stuck in training due to small improving. Halting...')
                                break

                            prev_train_loss = loss

                        if i % validation_interval == 0:
                            # TODO: Track training here and not above
                            sum_str, loss, acc, step_value = sess.run(
                                [val_summary_op, val_loss, val_acc, step]
                            )
                            val_writer.add_summary(sum_str, step_value)
                            print('[%d] Validation loss: %f, Accuracy: %f'
                                  % (step_value, loss, acc))

                            # Track best model
                            if best_validation['loss'] > loss:
                                print('[%d] New best found' % step_value)
                                save_model(sess, saver, val_writer_path, step_value)
                                best_validation = {
                                    'loss': loss, 'acc': acc, 'step': step_value
                                }
                    break

                print('Best model found at {}'.format(best_validation))

                coord.request_stop()
                coord.join(threads)

    def predict(self, X, y, **params):
        return None


# TODO: static method, could we moved outside
def get_loss_op(logits, y, sum_collection, **params):
    num_classes = logits.get_shape().as_list()[-1]
    l1_ratio = params.get('l1_ratio', None)
    l2_ratio = params.get('l2_ratio', None)

    l1_term = l1_norm(get_model_weights()) * l1_ratio \
        if l1_ratio is not None else tf.constant(0.0)
    tf.summary.scalar('l1_term', l1_term, [sum_collection])

    l2_term = l2_norm(get_model_weights()) * l2_ratio \
        if l2_ratio is not None else tf.constant(0.0)
    tf.summary.scalar('l2_term', l2_term, [sum_collection])

    loss_term = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.one_hot(y, depth=num_classes)
        )
    )

    tf.summary.scalar('loss_term', loss_term, [sum_collection])

    loss_op = loss_term + l1_term + l2_term
    tf.summary.scalar('total_loss', loss_op, [sum_collection])

    return loss_op


def build_workflow(dataset, reader, data_mode, step, reuse=False, **params):
    network_fn = params.get('network_fn', kernel_example_layout_fn)
    batch_size = params.get('batch_size', 128)
    memory_factor = params.get('memory_factor', 1)
    n_threads = params.get('n_threads', 2)

    features, labels = reader.read_batch(
        batch_size=batch_size,
        data_mode=DataMode.TRAINING,
        memory_factor=memory_factor,
        reader_threads=n_threads,
        train_mode=True
    )

    scope_params = {'reuse': reuse}
    with tf.variable_scope("network", **scope_params):
        logits = network_fn(features,
                            columns=dataset.get_wide_columns(),
                            outputs=dataset.get_num_classes())
        prediction = tf.nn.softmax(logits)

        loss_op = get_loss_op(
            logits=logits, y=labels, sum_collection=data_mode, **params
        )

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss_op, global_step=step)

        # Evaluate model
        accuracy_op = get_accuracy(prediction, labels)
        tf.summary.scalar('accuracy', accuracy_op, [data_mode])

    return logits, loss_op, train_op, accuracy_op


if __name__ == '__main__':

    a = DeepKernelModel()
    a.fit(data_settings_fn=AusSettings,
          folder='/media/walle/815d08cd-6bee-4a13-b6fd-87ebc1de2bb0/walle/kernel',  # noqa
          l2_ratio=0.01,
          data_location=get_data_location(Datasets.AUS))
