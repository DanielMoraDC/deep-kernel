import logging

import tensorflow as tf

from layout import get_layer_id, BATCH_NORM_COLLECTION
from kernels import KERNEL_COLLECTION

logger = logging.getLogger(__name__)


def get_all_variables(layers, include_output=True):
    ws_and_bs = _get_weights_and_biases(layers, include_output=include_output)
    kernels = _get_kernel_maps(layers)
    bns = _get_bn_vars(layers)
    return ws_and_bs + kernels + bns


def get_trainable_params(layers, include_output=True):
    selected = []
    for var in tf.trainable_variables():
        try:
            layer_name = var.name.split('/')[1]
            layer_id = get_layer_id(layer_name)
            if int(layer_id) in layers:
                selected.append(var)
        except Exception:
            # If we are here we assume we have an output variable
            if include_output:
                selected.append(var)
    return selected


def summarize_gradients(gradients, tag):
    for grad, var in gradients:
        if grad is not None:
            var_str_list = var.name.split('/')[1:]
            # Lets remove the :num ids
            var_str_list = [x.replace(':', '_') for x in var_str_list]
            var_id = '_'.join(var_str_list)
            tf.summary.histogram(var_id + "_gradient", grad, [tag])
            tf.summary.histogram(
                var_id + "_gradient_norm", tf.global_norm([grad]), [tag]
            )


def get_model_weights(layers, include_output=True):
    weights = []
    vars_layers = _get_weights_and_biases(layers, include_output)
    for var in vars_layers:
        if 'bias' in var.name:
            continue
        elif 'weight' in var.name:
            weights.append(var)
        else:
            raise RuntimeError('Unknown variable %s' % var.name)
    return weights


def _get_bn_vars(layers):
    selected = []
    bns = tf.get_collection(BATCH_NORM_COLLECTION)
    for var in bns:
        layer_name = var.name.split('/')[1]
        layer_id = get_layer_id(layer_name)
        if int(layer_id) in layers:
            selected.append(var)
    return selected


def _get_kernel_maps(layer_list):
    kernel_maps = tf.get_collection(KERNEL_COLLECTION)
    selected = []
    for km in kernel_maps:
        layer_name = km.name.split('/')[1]
        layer_id = get_layer_id(layer_name)
        if int(layer_id) in layer_list:
            selected.append(km)
    return selected


def _get_weights_and_biases(layer_list, include_output=True):
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
