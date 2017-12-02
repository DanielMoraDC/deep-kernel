import tensorflow as tf
import os
import logging
import numpy as np

from training import eval_epoch, run_training_epoch, build_run_context, \
    EarlyStop, CyclicPolicy
from ops import create_global_step, save_model, init_kernel_ops
from visualization import get_writer, write_epoch, write_scalar

from protodata.data_ops import DataMode
from protodata.reading_ops import DataReader

logger = logging.getLogger(__name__)

# TODO: switch Monitored Session by custom

# Disable Tensorflow debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DeepKernelModel():

    """ Class that combines fully connected and kernel functions """

    def __init__(self, verbose=True):
        self._verbose = verbose
        self._layer_idx = None
        self._epochs = None

    def fit(self, **params):

        # Mandatory parameters
        max_epochs = int(params.get('max_epochs'))
        data_settings_fn = params.get('data_settings_fn')
        data_location = params.get('data_location')

        # Parameters with default values
        folder = params.get('folder')
        is_layerwise = params.get('switch_epochs') is not None
        switch_epochs = params.get('switch_epochs').copy() \
            if is_layerwise else None
        summary_epochs = params.get('summary_epochs', 1)

        with tf.Graph().as_default() as graph:

            step = create_global_step()

            dataset = data_settings_fn(dataset_location=data_location)
            reader = DataReader(dataset)

            # Get training operations
            train_flds = range(dataset.get_fold_num())
            train_context = build_run_context(
                dataset, reader, DataMode.TRAINING, train_flds, step, **params
            )

            # Initialize writers and summaries
            writer = tf.summary.FileWriter(folder, graph)
            hist_summary_op = tf.summary.merge_all(DataMode.TRAINING)
            saver = tf.train.Saver()

            self._initialize_training(is_layerwise)

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
                        sess, train_context, self._layer_idx
                    )

                    if epoch % summary_epochs == 0:
                        # Store training summaries and log
                        sum_str = sess.run(hist_summary_op)
                        writer.add_summary(sum_str, epoch)
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

        return model_path

    def fit_and_validate(self, **params):

        # Mandatory parameters
        train_flds = params.get('training_folds')
        val_folds = params.get('validation_folds')
        max_epochs = int(params.get('max_epochs'))
        data_settings_fn = params.get('data_settings_fn')
        data_location = params.get('data_location')

        # Parameters with default values
        folder = params.get('folder', None)
        should_save = folder is not None
        strip_length = params.get('strip_length', 5)
        progress_thresh = params.get('progress_thresh', 0.1)
        max_successive_strips = params.get('max_successive_strips', 3)
        is_layerwise = params.get('layerwise', False)

        self._initialize_training(is_layerwise, **params)

        with tf.Graph().as_default() as graph:

            step = create_global_step()

            dataset = data_settings_fn(dataset_location=data_location)
            reader = DataReader(dataset)

            # Get training operations
            train_context = build_run_context(
                dataset, reader, DataMode.TRAINING, train_flds, step, **params
            )
            early_stop = EarlyStop(
                'global', progress_thresh, max_successive_strips
            )

            # Get validation operations
            val_context = build_run_context(
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

                init_kernel_ops(sess)

                # Define coordinator to handle all threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                for epoch in range(max_epochs):

                    epoch_loss, epoch_acc, epoch_l2 = run_training_epoch(
                        sess, train_context, self._layer_idx
                    )
                    early_stop.epoch_update(1-epoch_acc)

                    if epoch % strip_length == 0 and epoch != 0:

                        # Track training stats
                        self.log_info(
                            '[%d] Training Loss: %f, Error: %f'
                            % (epoch, epoch_loss, 1 - epoch_acc)
                        )

                        # Track validation stats
                        mean_val_loss, mean_val_acc, mean_val_l2 = eval_epoch(
                            sess, val_context, self._layer_idx
                        )
                        self.log_info(
                            '[%d] Validation loss: %f, Error: %f'
                            % (epoch, mean_val_loss, 1 - mean_val_acc)
                        )

                        if should_save:
                            # Write histograms
                            sum_str = sess.run(train_summary_op)
                            sum_str_val = sess.run(val_summary_op)
                            train_writer.add_summary(sum_str, epoch)
                            val_writer.add_summary(sum_str_val, epoch)
                            # Write learning rate
                            lr_val = sess.run(train_context.lr_op)
                            write_scalar(
                                train_writer, 'lr', lr_val, epoch
                            )
                            # Write stats
                            write_epoch(
                                train_writer, epoch_loss, epoch_acc, epoch_l2, epoch  # noqa
                            )
                            write_epoch(
                                val_writer, mean_val_loss, mean_val_acc, mean_val_l2, epoch  # noqa
                            )

                        is_best, stop, train_errors = early_stop.strip_update(
                            1 - epoch_acc,
                            epoch_loss,
                            1 - mean_val_acc,
                            mean_val_loss,
                            epoch
                        )

                        if is_best and should_save:
                            save_model(sess, saver, folder, epoch)

                        if stop and is_layerwise:
                            self._iterate_layer(epoch, train_errors)
                            early_stop.set_zero_fails()
                            if self._policy.cycle_ended():
                                _, l_stop, _ = self._layer_stop.strip_update(
                                    1 - epoch_acc,
                                    epoch_loss,
                                    1 - mean_val_acc,
                                    mean_val_loss,
                                    epoch
                                )
                                if l_stop:
                                    break

                        elif stop and not is_layerwise:
                            break

                best_model = early_stop.get_best()
                self.log_info('Best model found: {}'.format(best_model))

                coord.request_stop()
                coord.join(threads)

        return best_model

    def predict(self, X=None, y=None, **params):

        data_settings_fn = params.get('data_settings_fn')
        data_location = params.get('data_location')
        folder = params.get('folder')
        store_summaries = params.get('summaries', True)

        with tf.Graph().as_default() as graph:

            step = create_global_step()

            dataset = data_settings_fn(dataset_location=data_location)
            reader = DataReader(dataset)

            # Get training operations
            test_context = build_run_context(
                dataset=dataset, reader=reader, tag=DataMode.TEST,
                folds=None, step=step, is_training=False, **params
            )

            if store_summaries:
                writer = get_writer(graph, folder, DataMode.TEST)
            summary_op = tf.summary.merge_all(DataMode.TEST)

            saver = tf.train.Saver()

            with tf.train.MonitoredTrainingSession(
                    save_checkpoint_secs=None,
                    save_summaries_steps=None,
                    save_summaries_secs=None) as sess:

                ckpt = tf.train.get_checkpoint_state(folder)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise ValueError('No model found in %s' % folder)

                # Define coordinator to handle all threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                losses, accs, finish = [], [], False

                while not finish:

                    try:
                        # Track loss and accuracy until queue exhausted
                        loss, acc, summary = sess.run(
                            [
                                test_context.loss_ops[0],  # Use all layers
                                test_context.acc_op,
                                summary_op
                            ]
                        )

                        losses.append(loss)
                        accs.append(acc)

                        if store_summaries:
                            writer.add_summary(summary)

                    except tf.errors.OutOfRangeError:
                        logger.info('Queue exhausted. Read all instances')
                        finish = True

                coord.request_stop()
                coord.join(threads)

        return {'loss': np.mean(losses), 'error': 1 - np.mean(accs)}

    def log_info(self, msg):
        if self._verbose:
            logger.info(msg)

    def _initialize_training(self, is_layerwise, **params):
        if is_layerwise:
            self._epochs = []
            layerwise_thresh = params.get('layerwise_progress_thresh', 0.1)
            layerwise_succ_strips = params.get('layer_successive_strips', 1)
            self._layer_stop = EarlyStop(
                'layerwise', layerwise_thresh, layerwise_succ_strips
            )
            self._policy = CyclicPolicy(params.get('num_layers'))
            self._layer_idx = self._policy.initial_layer_id()
        else:
            self._layer_idx = 0

    '''
    def _initialize_training(self, layerwise):
        if layerwise:
            self._layer_idx = 1
            self._epochs = []
            self._train_errors = []
            self._prev_val_error = float('inf')
        else:
            self._layer_idx = 0
    '''

    def _iterate_layer(self, epoch, train_errors):
        # self._layer_idx = max(((self._layer_idx + 1) % (layers + 1)), 1)
        self._layer_idx = self._policy.next_layer_id()
        self._layer_stop.epoch_update(np.mean(train_errors))
        self._epochs.append(epoch)
        # self._train_errors.append(np.mean(train_errors))
        self.log_info('Switching to layer %d' % self._layer_idx)
