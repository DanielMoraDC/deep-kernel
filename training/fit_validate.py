import tensorflow as tf
import os
import logging

from sklearn.base import BaseEstimator, ClassifierMixin

from training.run_ops import eval_epoch, run_training_epoch, \
                             build_run_context, image_spec_from_params
from training.predict import predict_fn
from validation.early_stop import EarlyStop

from ops import get_global_step, save_model, init_kernel_ops
from visualization import get_writer, write_epoch, write_scalar

from protodata.data_ops import DataMode
from protodata.reading_ops import DataReader

logger = logging.getLogger(__name__)


# Disable Tensorflow debug messages
tf.logging.set_verbosity(tf.logging.ERROR)


class DeepNetworkValidation(BaseEstimator, ClassifierMixin):

    def __init__(self, settings_fn, data_location, folder=None):
        self._folder = folder
        self._settings_fn = settings_fn
        self._data_location = data_location
        self._train_writer, self._val_writer = None, None
        self._epochs = None

    def _init_writers(self, graph):
        self._train_writer = get_writer(
            graph, self._folder, DataMode.TRAINING
        )
        val_writer_path = os.path.join(self._folder, DataMode.VALIDATION)
        os.makedirs(val_writer_path)
        self._val_writer = tf.summary.FileWriter(val_writer_path, graph)

    def _initialize_training(self, **params):
        switch_policy_fn = params.get('switch_policy')
        self._epochs = []
        self._policy = switch_policy_fn(**params)
        self._layer_idx = self._policy.layer()
        logger.debug('Starting layerwise fit with %s' % self._policy.name())

    def _should_save(self):
        return self._folder is not None

    def _epoch_summary(self,
                       sess,
                       train_context,
                       train_run,
                       val_context,
                       val_run,
                       epoch):
        # Write histograms
        sum_str = sess.run(
            train_context.summary_op,
            feed_dict={train_context.is_training_op: True}
        )
        sum_str_val = sess.run(
            val_context.summary_op,
            feed_dict={val_context.is_training_op: False}
        )

        self._train_writer.add_summary(sum_str, epoch)
        self._val_writer.add_summary(sum_str_val, epoch)

        # Write learning rate
        lr_val = sess.run(train_context.lr_op)
        write_scalar(
            self._train_writer, 'lr', lr_val, epoch
        )

        # Write epoch statistics
        write_epoch(self._train_writer, train_run, epoch)
        write_epoch(self._val_writer, val_run, epoch)

    def fit(self, train_folds, val_folds, max_epochs, **params):

        max_epochs = int(max_epochs)
        epochs_per_layer = params.get('epochs_per_layer')

        # Parameters with default values
        strip_length = params.get('strip_length', 5)
        progress_thresh = params.get('progress_thresh', 0.1)
        max_successive_strips = params.get('max_successive_strips', 3)

        self._initialize_training(**params)

        with tf.Graph().as_default() as graph:

            step = get_global_step()

            dataset = self._settings_fn(
                dataset_location=self._data_location,
                image_specs=image_spec_from_params(**params)
            )
            reader = DataReader(dataset)

            # Get training operations
            train_context = build_run_context(
                dataset, reader, DataMode.TRAINING, train_folds, step, **params
            )
            early_stop = EarlyStop(
                'global', progress_thresh, max_successive_strips
            )

            # Get validation operations
            val_context = build_run_context(
                dataset, reader, DataMode.VALIDATION, val_folds, step, True, **params  # noqa
            )

            if self._should_save():
                self._init_writers(graph)
                saver = tf.train.Saver()

            with tf.train.MonitoredTrainingSession(
                    save_checkpoint_secs=None,
                    save_summaries_steps=None,
                    save_summaries_secs=None) as sess:

                init_kernel_ops(sess)

                # Define coordinator to handle all threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                while(True):

                    if sess.run(step) >= max_epochs:
                        logger.debug('Max epochs %d reached' % max_epochs)
                        break

                    train_run = run_training_epoch(
                        sess, train_context, self._layer_idx
                    )
                    early_stop.epoch_update(train_run.error())

                    epoch = train_run.epoch

                    if epoch % strip_length == 0:

                        # Track training stats
                        logger.debug(
                            '[%d] Training Loss: %f, Error: %f. L2: %f'
                            % (epoch, train_run.loss(), train_run.error(), train_run.l2())  # noqa
                        )

                        # Track validation stats
                        val_run = eval_epoch(
                            sess, val_context, self._layer_idx
                        )
                        logger.debug(
                            '[%d] Validation loss: %f, Error: %f'
                            % (epoch, val_run.loss(), val_run.error())
                        )

                        if self._should_save():
                            self._epoch_summary(
                                sess,
                                train_context,
                                train_run,
                                val_context,
                                val_run,
                                epoch
                            )

                        is_best, stop, train_errors = early_stop.strip_update(
                            train_run, val_run, epoch
                        )

                        if is_best and self._should_save():
                            save_model(sess, saver, self._folder, epoch)

                        if stop:
                            break

                    if epoch % epochs_per_layer == 0:
                        # Change layer
                        self._layer_idx = self._policy.next_layer_id()
                        logger.debug('Switching to layer %d' % self._layer_idx)

                best_model = early_stop.get_best()
                logger.debug('Best model found: {}'.format(best_model))

                coord.request_stop()
                coord.join(threads)

        return best_model

    def predict(self, **params):
        return predict_fn(
            self._settings_fn, self._data_location, self._folder, **params
        )
