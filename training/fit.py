import tensorflow as tf
import os
import logging
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from training.run_ops import RunStatus
from training import eval_epoch, run_training_epoch, build_run_context, \
    EarlyStop, RandomPolicy
from ops import create_global_step, save_model, init_kernel_ops
from visualization import get_writer, write_epoch, write_scalar

from training.predict import predict_fn

from protodata.data_ops import DataMode
from protodata.reading_ops import DataReader

logger = logging.getLogger(__name__)

# TODO: switch Monitored Session by custom

# Disable Tensorflow debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DeepNetworkTraining(BaseEstimator, ClassifierMixin):

    def __init__(self, folder, settings_fn, data_location):
        self._folder = folder
        self._settings_fn = settings_fn
        self._data_location = data_location

    def _initialize_fit(self, is_layerwise, **params):
        if is_layerwise:
            self._layer_idx = 0  # TODO, defined by swithcing policy
            logger.info(
                'Layerwise fit initialized...'
            )
        else:
            self._layer_idx = 0

    def _iterate_layer(self, epoch, train_errors):
        self._layer_idx = 1 # TODO fix
        self.log_info('Switching to layer %d' % self._layer_idx)

    def fit(self, folder, max_epochs, **params):

        # Parameters with default values
        is_layerwise = params.get('switch_epochs') is not None
        switch_epochs = params.get('switch_epochs').copy() \
            if is_layerwise else None
        summary_epochs = params.get('summary_epochs', 1)

        with tf.Graph().as_default() as graph:

            step = create_global_step()

            dataset = self._settings_fn(dataset_location=self._data_location)
            reader = DataReader(dataset)

            # Get training operations
            train_flds = range(dataset.get_fold_num())
            context = build_run_context(
                dataset, reader, DataMode.TRAINING, train_flds, step, **params
            )

            # Initialize writers and summaries
            writer = tf.summary.FileWriter(folder, graph)
            saver = tf.train.Saver()

            self._initialize_training(is_layerwise, **params)

            status = RunStatus()

            with tf.train.MonitoredTrainingSession(
                    save_checkpoint_secs=None,
                    save_summaries_steps=None,
                    save_summaries_secs=None) as sess:

                init_kernel_ops(sess)

                # Define coordinator to handle all threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                for epoch in range(max_epochs):

                    loss_mean, acc_mean, l2_mean = run_training_epoch(
                        sess, context, self._layer_idx
                    )

                    status.update(loss_mean, acc_mean, l2_mean)

                    if epoch % summary_epochs == 0:
                        # Store histogram
                        sum_str = sess.run(context.summary_op)
                        writer.add_summary(sum_str, epoch)

                        # Store stats from current epoch
                        write_epoch(
                            writer, loss_mean, acc_mean, l2_mean, epoch
                        )

                        self.log_info(
                            '[%d] Training Loss: %f, Error: %f'
                            % (epoch, loss_mean, 1-acc_mean)
                        )

                    if switch_epochs is not None and len(switch_epochs) > 0 \
                            and epoch == switch_epochs[0]:
                        self._iterate_layer(epoch, [1-acc_mean])
                        switch_epochs = switch_epochs[1:]

                self.log_info('Finished training at step %d' % max_epochs)
                model_path = save_model(sess, saver, folder, max_epochs)

                coord.request_stop()
                coord.join(threads)

        return model_path, status.get_means()

    def predict(self):
        return predict_fn(
            self._settings_fn, self._data_location, self._folder
        )
