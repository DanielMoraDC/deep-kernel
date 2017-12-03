import os

import tensorflow as tf


def get_writer(graph, folder, data_mode):
    writer_path = os.path.join(folder, data_mode)
    os.makedirs(writer_path)
    return tf.summary.FileWriter(writer_path, graph)


def write_epoch(writer, run_status, epoch):
    summary = tf.Summary(
        value=[
            tf.Summary.Value(tag='loss', simple_value=run_status.loss()),
            tf.Summary.Value(tag='error', simple_value=run_status.error()),
            tf.Summary.Value(tag='L2 term', simple_value=run_status.l2())
        ]
    )
    writer.add_summary(summary, epoch)


def write_scalar(writer, name, value, epoch):
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=name, simple_value=value)]
    )
    writer.add_summary(summary, epoch)
