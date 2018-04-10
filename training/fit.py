import tensorflow as tf
import os
import logging

from sklearn.base import BaseEstimator, ClassifierMixin

from training.run_ops import run_training_epoch, build_run_context, \
                             image_spec_from_params
from training.predict import predict_fn

from variables import get_all_variables
from ops import save_model, init_kernel_ops, get_global_step
from visualization import write_epoch

from protodata.data_ops import DataMode
from protodata.reading_ops import DataReader

logger = logging.getLogger(__name__)


# Disable Tensorflow debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DeepNetworkTraining(BaseEstimator, ClassifierMixin):

    def __init__(self, folder, settings_fn, data_location):
        self._folder = folder
        self._settings_fn = settings_fn
        self._data_location = data_location
        self._aux_saver, self._restore_vars = None, None

    def _initialize_fit(self, is_layerwise, **params):
        if is_layerwise:
            switch_policy_fn = params.get('switch_policy')
            self._policy = switch_policy_fn(**params)
            self._layer_idx = self._policy.layer()
            logger.debug(
                'Layerwise fit initialized...'
            )
        else:
            self._layer_idx = params.get('train_only', 0)

    def _init_session(self, sess, **params):
        init_kernel_ops(sess)

        # If folder provided, restore variables
        restore_folder = params.get('restore_folder')
        if restore_folder is not None:
            ckpt = tf.train.get_checkpoint_state(restore_folder)
            if ckpt and ckpt.model_checkpoint_path:
                logger.debug(
                    "Restoring {} variables from {}".format(
                        self._restore_vars,
                        ckpt.model_checkpoint_path
                    )
                )
                self._aux_saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise ValueError('No model found in %s' % restore_folder)
        else:
            logger.debug("Starting model from scratch")

    def _init_savers(self, step, **params):
        saver = tf.train.Saver()
        if params.get('restore_folder', None) is not None:
            self._restore_vars = get_all_variables(
                params.get('restore_layers'),
                include_output=False
            )
            self._restore_vars.append(step)
            self._aux_saver = tf.train.Saver(self._restore_vars)
        return saver

    def fit(self, max_epochs, **params):

        max_epochs = int(max_epochs)

        # Parameters with default values
        is_layerwise = params.get('switch_epochs') is not None
        switch_epochs = params.get('switch_epochs').copy() \
            if is_layerwise else None
        summary_epochs = params.get('summary_epochs', 1)

        with tf.Graph().as_default() as graph:

            step = get_global_step()

            dataset = self._settings_fn(
                dataset_location=self._data_location,
                image_specs=image_spec_from_params(**params)
            )
            reader = DataReader(dataset)

            # Get training operations
            train_flds = range(dataset.get_fold_num())
            context = build_run_context(
                dataset, reader, DataMode.TRAINING, train_flds, step, **params
            )

            # Initialize writers and summaries
            writer = tf.summary.FileWriter(self._folder, graph)

            saver = self._init_savers(step, **params)

            self._initialize_fit(is_layerwise, **params)

            with tf.train.MonitoredTrainingSession(
                    save_checkpoint_secs=None,
                    save_summaries_steps=None,
                    save_summaries_secs=None) as sess:

                self._init_session(sess, **params)

                # Define coordinator to handle all threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                while(True):

                    if sess.run(step) >= max_epochs:
                        logger.debug('Max epochs %d reached' % max_epochs)
                        break

                    run = run_training_epoch(
                        sess, context, self._layer_idx
                    )

                    epoch = run.epoch

                    if epoch % summary_epochs == 0:
                        # Store histogram
                        sum_str = sess.run(
                            context.summary_op,
                            feed_dict={context.is_training_op: True}
                        )
                        writer.add_summary(sum_str, epoch)

                        # Store stats from current epoch
                        write_epoch(writer, run, epoch)

                        logger.debug(
                            '[%d] Training Loss: %f, Error: %f. L2: %f'
                            % (epoch, run.loss(), run.error(), run.l2())
                        )

                    if switch_epochs is not None and len(switch_epochs) > 0 \
                            and epoch == switch_epochs[0]:
                        self._layer_idx = self._policy.next_layer_id()
                        logger.debug(
                            'Switching to layer %d' % self._layer_idx
                        )
                        switch_epochs = switch_epochs[1:]

                logger.debug('Finished training at step %d' % max_epochs)
                model_path = save_model(sess, saver, self._folder, max_epochs)

                coord.request_stop()
                coord.join(threads)

        return model_path, run.loss(), run.error(), run.l2()

    def predict(self, **params):
        return predict_fn(
            self._settings_fn, self._data_location, self._folder, **params
        )
