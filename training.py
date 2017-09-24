import tensorflow as tf


def get_global_step(graph=None):
    """ Loads the unique global step, if found """
    graph = tf.get_default_graph() if graph is None else graph
    global_step_tensors = graph.get_collection(tf.GraphKeys.GLOBAL_STEP)
    if len(global_step_tensors) == 0:
        raise RuntimeError('No global step stored in the collection')
    elif len(global_step_tensors) > 1:
        raise RuntimeError('Multiple instances found for the global step')
    return global_step_tensors[0]


def create_global_step():
    """ Creates a global step in the VARIABLEs and GLOBAL_STEP collections """
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP]
    return tf.get_variable('global_step', shape=[],
                           dtype=tf.int32,
                           initializer=tf.constant_initializer(0),
                           trainable=False,
                           collections=collections)
