from sklearn.base import RegressorMixin
import tensorflow as tf
import os
import logging

from training import create_global_step, get_model_weights, l1_norm, \
    l2_norm, get_accuracy, save_model, get_writer
from layout import example_layout_fn, kernel_example_layout_fn

from protodata.data_ops import DataMode
from protodata.reading_ops import DataReader
from protodata.datasets import AusSettings
from protodata.utils import get_data_location
from protodata.datasets import Datasets

logger = logging.getLogger(__name__)

# Disable Tensorflow debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DeepKernelModel(RegressorMixin):

    """ Class that combines fully connected and kernel functions """

    def __init__(self, verbose=True):
        self._verbose = verbose

    def fit(self, X=None, y=None, **params):

        train_folds = params.get('training_folds', None)
        val_folds = params.get('validation_folds', None)
        max_steps = params.get('max_steps', None)

        if train_folds is not None and val_folds is not None \
                and max_steps is not None:
            return self.fit_and_validate(**params)
        elif max_steps is not None:
            return self.fit_training(**params)
        else:
            raise ValueError(
                'Either training and validation folds or ' +
                'training folds and steps must be provided'
            )

    def fit_training(self, **params):

        # Mandatory parameters
        steps = int(params.get('max_steps'))
        data_settings_fn = params.get('data_settings_fn')
        data_location = params.get('data_location')

        # Parameters with default values
        folder = params.get('folder')
        summaries_steps = params.get('summaries_steps', 50)

        with tf.Graph().as_default() as graph:

            step = create_global_step()  # noqa

            dataset = data_settings_fn(dataset_location=data_location)
            reader = DataReader(dataset)

            # Get training operations
            train_flds = range(dataset.get_fold_num())
            _, loss_op, train_op, acc_op = build_workflow(
                dataset, reader, DataMode.TRAINING, train_flds, step, **params
            )

            # Initialize writers and summaries
            writer_path = os.path.join(folder, DataMode.TRAINING)
            os.makedirs(writer_path)
            writer = tf.summary.FileWriter(writer_path, graph)
            summary_op = tf.summary.merge_all(DataMode.TRAINING)
            saver = tf.train.Saver()

            with tf.train.MonitoredTrainingSession(
                    save_checkpoint_secs=None,
                    save_summaries_steps=None,
                    save_summaries_secs=None) as sess:

                # Define coordinator to handle all threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                while not sess.should_stop():

                    for i in range(steps):

                        sess.run([train_op])

                        if i % summaries_steps == 0:
                            # Store training summaries
                            sum_str, step_value = sess.run(
                                [summary_op, step]
                            )
                            writer.add_summary(sum_str, step_value)

                            # Track training loss
                            sum_str, loss, acc, step_value = sess.run(
                                [summary_op, loss_op, acc_op, step]
                            )

                            self.log_info(
                                '[%d] Training Loss: %f, Accuracy: %f'
                                % (step_value, loss, acc)
                            )

                    break

                self.log_info('Finished training at step %d' % steps)
                model_path = save_model(sess, saver, writer_path, steps)

                coord.request_stop()
                coord.join(threads)

        return model_path

    def fit_and_validate(self, **params):

        # Mandatory parameters
        train_flds = params.get('training_folds')
        val_folds = params.get('validation_folds')
        max_steps = params.get('max_steps')
        data_settings_fn = params.get('data_settings_fn')
        data_location = params.get('data_location')

        # Parameters with default values
        folder = params.get('folder', None)
        should_save = folder is not None
        summaries_steps = params.get('summaries_steps', 50)
        validation_interval = params.get('validation_interval', 500)
        train_tolerance = params.get('train_tolerance', 1e-3)

        # Tracking initialization
        prev_train_loss = float('inf')
        best_validation = {
            'val_loss': float('inf'), 'val_acc': None, 'step': 0,
            'train_loss': None, 'train_acc': None
        }

        with tf.Graph().as_default() as graph:

            step = create_global_step()  # noqa

            dataset = data_settings_fn(dataset_location=data_location)
            reader = DataReader(dataset)

            # Get training operations
            _, train_loss_op, train_op, train_acc_op = build_workflow(
                dataset, reader, DataMode.TRAINING, train_flds, step, **params
            )

            # Get validation operations
            _, val_loss_op, _, val_acc_op = build_workflow(
                dataset, reader, DataMode.VALIDATION, val_folds, step, True, **params  # noqa
            )

            if should_save:
                # Initialize writers
                train_writer = get_writer(graph, folder, DataMode.TRAINING)

                val_writer_path = os.path.join(folder, DataMode.VALIDATION)
                os.makedirs(val_writer_path)
                val_writer = tf.summary.FileWriter(val_writer_path, graph)

            # Gather summaries by dataset
            train_summary_op = tf.summary.merge_all(DataMode.TRAINING)
            val_summary_op = tf.summary.merge_all(DataMode.VALIDATION)

            if should_save:
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

                        if should_save and i % summaries_steps == 0:
                            # Store training summaries
                            sum_str, step_value = sess.run(
                                [train_summary_op, step]
                            )
                            train_writer.add_summary(sum_str, step_value)

                        if i % validation_interval == 0:

                            # Track training loss
                            train_loss, train_acc, step_value = sess.run(
                                [train_loss_op, train_acc_op, step]
                            )
                            self.log_info(
                                '[%d] Training Loss: %f, Accuracy: %f'
                                % (step_value, train_loss, train_acc)
                            )

                            # Track validation loss
                            sum_str, val_loss, val_acc = sess.run(
                                [val_summary_op, val_loss_op, val_acc_op]
                            )

                            if should_save:
                                val_writer.add_summary(sum_str, step_value)

                            self.log_info(
                                '[%d] Validation loss: %f, Accuracy: %f'
                                % (step_value, val_loss, val_acc)
                            )

                            # Track best model at validation
                            if best_validation['val_loss'] > val_loss:
                                self.log_info(
                                    '[%d] New best found' % step_value
                                )

                                if should_save:
                                    save_model(sess,
                                               saver,
                                               val_writer_path,
                                               step_value)

                                best_validation = {
                                    'val_loss': val_loss,
                                    'val_acc': val_acc,
                                    'step': step_value,
                                    'train_loss': train_loss,
                                    'train_acc': train_acc
                                }

                            # Stop if training hasn't improved much
                            improved_diff = (prev_train_loss - train_loss)
                            improved_ratio = improved_diff / prev_train_loss
                            if train_loss > prev_train_loss or \
                                    improved_ratio < train_tolerance:
                                self.log_info(
                                    'Stuck in training due to small ' +
                                    'or no improving. Halting...'
                                )
                                break
                            prev_train_loss = train_loss

                    break

                self.log_info('Best model found: {}'.format(best_validation))

                coord.request_stop()
                coord.join(threads)

                return best_validation

    def predict(self, X, y, **params):
        return None

    def log_info(self, msg):
        if self._verbose:
            logger.info(msg)


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


def build_workflow(dataset,
                   reader,
                   data_mode,
                   folds,
                   step,
                   reuse=False,
                   **params):
    lr = params.get('lr', 1e-2)
    network_fn = params.get('network_fn', kernel_example_layout_fn)
    batch_size = params.get('batch_size', 32)
    memory_factor = params.get('memory_factor', 1)
    n_threads = params.get('n_threads', 2)

    features, labels = reader.read_folded_batch(
        batch_size=batch_size,
        data_mode=DataMode.TRAINING,
        folds=folds,
        memory_factor=memory_factor,
        reader_threads=n_threads,
        train_mode=True,
        shuffle=True
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

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss_op, global_step=step)

        # Evaluate model
        accuracy_op = get_accuracy(prediction, labels)
        tf.summary.scalar('accuracy', accuracy_op, [data_mode])

    return logits, loss_op, train_op, accuracy_op


if __name__ == '__main__':

    a = DeepKernelModel()
    a.fit(data_settings_fn=AusSettings,
          folder='/media/walle/815d08cd-6bee-4a13-b6fd-87ebc1de2bb0/walle/kernel',  # noqa
          training_folds=[0, 1, 2, 3, 4, 5, 6, 7, 8],
          validation_folds=[9],
          l2_ratio=0.01,
          data_location=get_data_location(Datasets.AUS, folded=True))
