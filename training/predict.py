import tensorflow as tf
import logging

from ops import create_global_step, save_model, init_kernel_ops
from visualization import get_writer, write_epoch, write_scalar
from training.run_ops import build_run_context, RunStatus

from protodata.data_ops import DataMode
from protodata.reading_ops import DataReader


logger = logging.getLogger(__name__)


def predict_fn(self, data_settings_fn, data_location, folder, **params):

    store_summaries = params.get('summaries', True)

    with tf.Graph().as_default() as graph:

        step = create_global_step()

        dataset = data_settings_fn(dataset_location=data_location)
        reader = DataReader(dataset)

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

            status, finish = RunStatus(), False

            while not finish:

                try:
                    # Track loss and accuracy until queue exhausted
                    loss, acc, l2, summary = sess.run(
                        [
                            test_context.loss_ops[0],  # Use all layers
                            test_context.acc_op,
                            test_context.l2_op,
                            summary_op
                        ]
                    )

                    status.update(loss, acc, l2)

                    if store_summaries:
                        writer.add_summary(summary)

                except tf.errors.OutOfRangeError:
                    logger.info('Queue exhausted. Read all instances')
                    finish = True

            coord.request_stop()
            coord.join(threads)

    loss_mean, acc_mean, _ = status.get_means()

    return {'loss': loss_mean, 'error': 1 - acc_mean}
