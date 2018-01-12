import logging

import tensorflow as tf

from layout import get_layer_id
from kernels import KERNEL_COLLECTION

logger = logging.getLogger(__name__)


def get_model_weights(layers, include_output=True):
    """ Returns the list of models parameters """
    weights = []
    vars_layers = get_weights_and_biases(layers, include_output)
    for var in vars_layers:
        if 'bias' in var.name:
            logger.debug('Ignoring bias %s for regularization ' % (var.name))
        elif 'weight' in var.name:
            weights.append(var)
            logger.debug('Using weights %s for regularization ' % (var.name))
        else:
            raise RuntimeError('Unknown variable %s' % var.name)
    return weights


def get_all_variables(layers, include_output=True):
    ws_and_bs = get_weights_and_biases(layers, include_output=include_output)
    kernels = get_kernel_maps(layers)
    return ws_and_bs + kernels


def get_kernel_maps(layer_list):
    kernel_maps = tf.get_collection(KERNEL_COLLECTION)
    selected = []
    for km in kernel_maps:
        layer_name = km.name.split('/')[1]
        layer_id = get_layer_id(layer_name)
        if int(layer_id) in layer_list:
            selected.append(km)
    return selected


def get_weights_and_biases(layer_list, include_output=True):
    selected = []
    ws_and_bs = tf.get_collection(tf.GraphKeys.WEIGHTS)
    for var in ws_and_bs:
        try:
            layer_name = var.name.split('/')[1]
            layer_id = get_layer_id(layer_name)
            if int(layer_id) in layer_list:
                selected.append(var)
        except Exception:
            # If we are here we assume we have an output variable
            if include_output:
                selected.append(var)
    return selected
