from sklearn.base import RegressorMixin
import tensorflow as tf
import os
import logging
import numpy as np

from training import create_global_step, get_model_weights, l1_norm, \
    l2_norm, get_accuracy_op, save_model, get_writer, eval_epoch, \
    progress, RunContext, run_training_epoch, get_loss_fn, \
    init_kernel_ops
from layout import example_layout_fn, kernel_example_layout_fn

from protodata.data_ops import DataMode
from protodata.reading_ops import DataReader

logger = logging.getLogger(__name__)

# TODO: switch Monitored Session by custom

# Disable Tensorflow debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DeepKernelModel(RegressorMixin):

    """ Class that combines fully connected and kernel functions """

    def __init__(self, verbose=True):
        self._verbose = verbose

    def fit(self, X=None, y=None, **params):

        train_folds = params.get('training_folds', None)
        val_folds = params.get('validation_folds', None)
        max_epochs = params.get('max_epochs', None)

        if train_folds is not None and val_folds is not None \
                and max_epochs is not None:
            return self.fit_and_validate(**params)
        elif max_epochs is not None:
            return self.fit_training(**params)
        else:
            raise ValueError(
                'Either training and validation folds or ' +
                'number of epochs steps must be provided'
            )

    def fit_training(self, **params):

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

                init_kernel_ops(sess)

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

            with tf.train.MonitoredTrainingSession(
                    save_checkpoint_secs=None,
                    save_summaries_steps=None,
                    save_summaries_secs=None) as sess:

                init_kernel_ops(sess)

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

    def predict(self, X=None, y=None, **params):

        data_settings_fn = params.get('data_settings_fn')
        data_location = params.get('data_location')
        folder = params.get('folder')

        with tf.Graph().as_default():

            step = create_global_step()

            dataset = data_settings_fn(dataset_location=data_location)
            reader = DataReader(dataset)

            # Get training operations
            test_context = build_run_context(
                dataset=dataset, reader=reader, tag=DataMode.TEST,
                folds=None, step=step, is_training=False, **params
            )

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
                        loss, acc = sess.run(
                            [test_context.loss_op, test_context.acc_op]
                        )
                        losses.append(loss)
                        accs.append(acc)

                    except tf.errors.OutOfRangeError:
                        logger.info('Queue exhausted. Read all instances')
                        finish = True

                coord.request_stop()
                coord.join(threads)

        return {'loss': np.mean(losses), 'error': 1 - np.mean(accs)}

    def log_info(self, msg):
        if self._verbose:
            logger.warn(msg)


def get_loss_op(logits, y, sum_collection, n_classes, **params):
    l1_ratio = params.get('l1_ratio', None)
    l2_ratio = params.get('l2_ratio', None)

    l1_term = l1_norm(get_model_weights()) * l1_ratio \
        if l1_ratio is not None else tf.constant(0.0)
    tf.summary.scalar('l1_term', l1_term, [sum_collection])

    l2_term = l2_norm(get_model_weights()) * l2_ratio \
        if l2_ratio is not None else tf.constant(0.0)
    tf.summary.scalar('l2_term', l2_term, [sum_collection])

    loss_term = tf.reduce_mean(
        get_loss_fn(
            logits, y, n_classes
        )
    )

    tf.summary.scalar('loss_term', loss_term, [sum_collection])

    loss_op = loss_term + l1_term + l2_term
    tf.summary.scalar('total_loss', loss_op, [sum_collection])

    return loss_op


def build_run_context(dataset,
                      reader,
                      tag,
                      folds,
                      step,
                      reuse=False,
                      is_training=True,
                      **params):
    lr = params.get('lr', 0.01)
    lr_decay = params.get('lr_decay', 0.5)
    lr_decay_epocs = params.get('lr_decay_epochs', 500)
    network_fn = params.get('network_fn', kernel_example_layout_fn)
    batch_size = params.get('batch_size')
    memory_factor = params.get('memory_factor')
    n_threads = params.get('n_threads')

    if folds is not None:
        fold_size = dataset.get_fold_size()
        steps_per_epoch = int(fold_size * len(folds) / batch_size)
        lr_decay_steps = lr_decay_epocs * steps_per_epoch
    else:
        steps_per_epoch = None
        lr_decay_steps = 10000  # Default value, not used

    data_subset = DataMode.TRAINING if tag == DataMode.VALIDATION else tag
    features, labels = reader.read_folded_batch(
        batch_size=batch_size,
        data_mode=data_subset,
        folds=folds,
        memory_factor=memory_factor,
        reader_threads=n_threads,
        train_mode=is_training,
        shuffle=True
    )

    scope_params = {'reuse': reuse}
    with tf.variable_scope("network", **scope_params):

        logits = network_fn(features,
                            columns=dataset.get_wide_columns(),
                            outputs=dataset.get_num_classes(),
                            tag=tag,
                            is_training=is_training,
                            **params)

        loss_op = get_loss_op(
            logits=logits,
            y=labels,
            sum_collection=tag,
            n_classes=dataset.get_num_classes(),
            **params
        )

        # Decaying learning rate: lr(t)' = lr / (1 + decay * t)
        decayed_lr = tf.train.inverse_time_decay(
            lr, step, decay_steps=lr_decay_steps, decay_rate=lr_decay
        )
        tf.summary.scalar('lr', decayed_lr, [tag])

        optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr)

        # This is needed for the batch norm moving averages
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss_op, global_step=step)

        # Evaluate model
        accuracy_op = get_accuracy_op(
            logits, labels, dataset.get_num_classes()
        )
        tf.summary.scalar('accuracy', accuracy_op, [tag])

    return RunContext(
        logits_op=logits,
        train_op=train_op,
        loss_op=loss_op,
        acc_op=accuracy_op,
        steps_per_epoch=steps_per_epoch
    )


from protodata import datasets
from protodata.utils import get_data_location
import sys
import shutil

if __name__ == '__main__':

    fit = bool(int(sys.argv[1]))
    
    folder = '/media/walle/815d08cd-6bee-4a13-b6fd-87ebc1de2bb0/walle/model'

    if fit:

        if os.path.isdir(folder):
            shutil.rmtree(folder)            

        '''
        m = DeepKernelModel(verbose=True)
        m.fit(
            data_settings_fn=datasets.AusSettings,
            training_folds=range(9),
            validation_folds=[9],
            max_epochs=100000,
            data_location=get_data_location(datasets.Datasets.AUS, folded=True),  # noqa
            l2_ratio=1e-2,
            lr=1e-4,
            memory_factor=2,
            hidden_units=64,
            n_threads=4,
            kernel_size=128,
            kernel_mean=0.0,
            kernel_std=0.01,
            strip_length=5,
            batch_size=32,
            folder=folder
        )
        '''

        m = DeepKernelModel(verbose=True)
        m.fit(
            data_settings_fn=datasets.AusSettings,
            training_folds=range(9),
            validation_folds=[9],
            max_epochs=100000,
            data_location=get_data_location(datasets.Datasets.AUS, folded=True),  # noqa
            l2_ratio=1e-4,
            lr=1e-4,
            lr_decay=0.5,
            lr_decay_epocs=128,
            memory_factor=2,
            hidden_units=128,
            n_threads=4,
            kernel_size=64,
            kernel_mean=0.0,
            kernel_std=0.1,
            strip_length=5,
            batch_size=16,
            folder=folder
        )

    else:

        m = DeepKernelModel(verbose=True)

        res = m.predict(
            data_settings_fn=datasets.AusSettings,
            folder=folder,
            data_location=get_data_location(datasets.Datasets.AUS, folded=True),  # noqa
            memory_factor=2,
            n_threads=4,
            hidden_units=128,
            kernel_size=64,
            kernel_mean=0.0,
            kernel_std=0.1,
            batch_size=16,
        )

        print('Got results {} for prediction'.format(res))
