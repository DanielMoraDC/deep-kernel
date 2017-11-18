import tensorflow as tf
import os
import logging
import numpy as np

from training import create_global_step, save_model, get_writer, \
    eval_epoch, progress, run_training_epoch, init_kernel_ops, \
    build_run_context, get_restore_info

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

    def fit(self, **params):

        # Mandatory parameters
        max_epochs = int(params.get('max_epochs'))
        data_settings_fn = params.get('data_settings_fn')
        data_location = params.get('data_location')

        # Parameters with default values
        folder = params.get('folder')
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
            summary_op = tf.summary.merge_all(DataMode.TRAINING)
            saver = tf.train.Saver()

            with tf.train.MonitoredTrainingSession(
                    save_checkpoint_secs=None,
                    save_summaries_steps=None,
                    save_summaries_secs=None) as sess:

                self._perform_assigns(sess, **params)

                # Define coordinator to handle all threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                for epoch in range(max_epochs):

                    loss_mean, acc_mean = run_training_epoch(
                        sess, train_context
                    )

                    if epoch % summary_epochs == 0:
                        # Store training summaries
                        sum_str = sess.run(summary_op)
                        writer.add_summary(sum_str, epoch)

                        self.log_info(
                            '[%d] Training Loss: %f, Accuracy: %f'
                            % (epoch, loss_mean, acc_mean)
                        )

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
        summary_epochs = params.get('summary_epochs', 1)
        strip_length = params.get('strip_length', 5)
        progress_thresh = params.get('progress_thresh', 0.1)
        max_successive_strips = params.get('max_successive_strips', 3)

        best_model = {'val_error': float('inf')}
        prev_val_err = float('inf')
        successive_fails = 0

        with tf.Graph().as_default() as graph:

            step = create_global_step()

            dataset = data_settings_fn(dataset_location=data_location)
            reader = DataReader(dataset)

            # Get training operations
            train_context = build_run_context(
                dataset, reader, DataMode.TRAINING, train_flds, step, **params
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

            train_losses, train_errors = [], []

            if should_save:
                saver = tf.train.Saver()

            if 'prev_layer_folder' in params:
                restore_info = get_restore_info(
                    params['num_layers'], params['prev_layer_folder']
                )
            else:
                restore_info = None

            with tf.train.MonitoredTrainingSession(
                    save_checkpoint_secs=None,
                    save_summaries_steps=None,
                    save_summaries_secs=None) as sess:

                self._perform_assigns(sess, restore_info, **params)

                # Define coordinator to handle all threads
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                for epoch in range(max_epochs):

                    epoch_loss, epoch_acc = run_training_epoch(
                        sess, train_context
                    )
                    train_losses.append(epoch_loss)
                    train_errors.append(1-epoch_acc)

                    if should_save and epoch % summary_epochs == 0:
                        # Store training summaries for current
                        sum_str = sess.run(train_summary_op)
                        sum_str_val = sess.run(val_summary_op)
                        train_writer.add_summary(sum_str, epoch)
                        val_writer.add_summary(sum_str_val, epoch)

                    if epoch % strip_length == 0 and epoch != 0:

                        # Track training loss and restart values
                        self.log_info(
                            '[%d] Training Loss: %f, Error: %f'
                            % (epoch, epoch_loss, 1 - epoch_acc)
                        )

                        # Track validation loss
                        mean_val_loss, mean_val_acc = eval_epoch(
                            sess, val_context
                        )
                        self.log_info(
                            '[%d] Validation loss: %f, Error: %f'
                            % (epoch, mean_val_loss, 1 - mean_val_acc)
                        )

                        # Track best model at validation
                        if best_model['val_error'] > (1 - mean_val_acc):
                            self.log_info('[%d] New best found' % epoch)

                            if should_save:
                                save_model(
                                    sess, saver, folder, epoch
                                )

                            best_model = {
                                'val_loss': mean_val_loss,
                                'val_error': 1 - mean_val_acc,
                                'epoch': epoch,
                                'train_loss': epoch_loss,
                                'train_error': 1 - epoch_acc,
                            }

                        # Stop using progress criteria
                        train_progress = progress(train_errors)
                        if train_progress < progress_thresh:
                            self.log_info(
                                '[%d] Stuck in training due to ' % epoch +
                                'lack of progress (%f < %f). Halting...'
                                % (train_progress, progress_thresh)
                            )
                            break

                        # Stop using UP criteria
                        if prev_val_err < (1 - mean_val_acc):
                            successive_fails += 1
                            self.log_info(
                                '[%d] Validation error increases. ' % epoch +
                                'Successive fails: %d' % successive_fails
                            )
                        else:
                            successive_fails = 0

                        if successive_fails == max_successive_strips:
                            self.log_info(
                                '[%d] Validation error increased ' % epoch +
                                'for %d successive ' % max_successive_strips +
                                'times. Halting ...'
                            )
                            break

                        # Restart strip information
                        prev_val_err = 1 - mean_val_acc
                        train_losses, train_errors = [], []

                self.log_info('Best model found: {}'.format(best_model))

                coord.request_stop()
                coord.join(threads)

                return best_model

    def _perform_assigns(self, sess, restore_info=None, **params):
        """
        Assigns variables from folder checkpoints and perform pending ops
        """
        self.log_info('Initializing kernels...')
        init_kernel_ops(sess)

        if restore_info is None:
            self.log_info(
                'No restore info provided. No assign ops will be performed'
            )
        else:
            restore_saver, prev_layer_folder, variables = restore_info
            ckpt = tf.train.get_checkpoint_state(prev_layer_folder)
            if ckpt and ckpt.model_checkpoint_path:
                self.log_info('Restoring variables {}'.format(variables))
                restore_saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise RuntimeError(
                    'Restore saver provided but no valid checkpoint ' +
                    'found in folder %s' % prev_layer_folder
                )

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
                                test_context.loss_op,
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
            logger.warn(msg)


from protodata import datasets
from protodata.utils import get_data_location
import sys
import shutil

if __name__ == '__main__':

    fit = bool(int(sys.argv[1]))
    
    folder = '/media/walle/815d08cd-6bee-4a13-b6fd-87ebc1de2bb0/walle/model'
    
    '''
    params = {
        'l2_ratio': 1e-3,
        'lr': 1e-3,
        'lr_decay': 0.5,
        'lr_decay_epocs': 400,
        'memory_factor': 2,
        'hidden_units': 64,
        'n_threads': 4,
        'kernel_size': 64,
        'kernel_mean': 0.0,
        'kernel_std': 0.1,
        'strip_length': 2,
        'batch_size': 16
    }
    '''

    params = {
        'l2_ratio': 1e-3,
        'lr': 1e-2,
        'lr_decay': 0.5,
        'lr_decay_epocs': 400,
        'memory_factor': 2,
        'hidden_units': 128,
        'n_threads': 4,
        'kernel_size': 128,
        'kernel_mean': 0.0,
        'kernel_std': 0.1,
        'strip_length': 2,
        'batch_size': 16
    }

    m = DeepKernelModel(verbose=True)

    if fit:

        if os.path.isdir(folder):
            shutil.rmtree(folder)            

        m.fit_and_validate(
            data_settings_fn=datasets.Monk2Settings,
            training_folds=range(9),
            validation_folds=[9],
            max_epochs=100000,
            data_location=get_data_location(datasets.Datasets.MONK2, folded=True),  # noqa
            folder=folder,
            **params
        )

    else:

        res = m.predict(
            data_settings_fn=datasets.Monk2Settings,
            folder=folder,
            data_location=get_data_location(datasets.Datasets.MONK2, folded=True),  # noqa
            **params
        )

        print('Got results {} for prediction'.format(res))
