import tensorflow as tf
import logging

from ops import get_global_step
from visualization import get_writer
from training.run_ops import build_run_context, test_step, RunStatus, \
                             image_spec_from_params
from protodata.data_ops import DataMode
from protodata.reading_ops import DataReader


logger = logging.getLogger(__name__)


def predict_fn(data_settings_fn, data_location, folder, **params):

    store_summaries = params.get('summaries', True)

    with tf.Graph().as_default() as graph:

        step = get_global_step()

        dataset = data_settings_fn(
            dataset_location=data_location,
            image_specs=image_spec_from_params(**params)
        )
        reader = DataReader(dataset)

        test_context = build_run_context(
            dataset=dataset,
            reader=reader,
            tag=DataMode.TEST,
            folds=None,
            step=step,
            **params
        )

        if store_summaries:
            writer = get_writer(graph, folder, DataMode.TEST)

        saver = tf.train.Saver()

        with tf.train.MonitoredTrainingSession(
                save_checkpoint_secs=None,
                save_summaries_steps=None,
                save_summaries_secs=None) as sess:

            ckpt = tf.train.get_checkpoint_state(folder)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                logger.debug(
                    'Restoring {} from {}'.format(
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES),
                        ckpt.model_checkpoint_path
                    )
                )
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
                    loss, acc, l2 = test_step(sess, test_context)
                    status.update(loss, acc, l2)

                    if store_summaries:
                        summary = sess.run(
                            test_context.summary_op,
                            feed_dict={test_context.is_training_op: False}
                        )
                        writer.add_summary(summary)

                except tf.errors.OutOfRangeError:
                    logger.info('Queue exhausted. Read all instances')
                    finish = True

            coord.request_stop()
            coord.join(threads)

    return {'loss': status.loss(), 'l2': status.l2(), 'error': status.error()}
