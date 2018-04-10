import abc

import numpy as np
import tensorflow as tf

KERNEL_COLLECTION = 'KERNEL_VARS'
KERNEL_ASSIGN_OPS = 'KERNEL_ASSIGN_OPS'


class RandomFourierFeatures(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, name, input_dims, kernel_size):
        self._name = name
        self._input_dims = input_dims
        self._kernel_size = kernel_size

    @abc.abstractmethod
    def draw_w(self):
        """
        Draws w samples according to the corresponding Fourier distribution
        """

    @abc.abstractmethod
    def draw_b(self):
        """
        Draws v samples according to the corresponding Fourier distribution
        """

    def apply_kernel(self, x, tag):

        w_name = '_'.join([self._name, 'w'])  # Name is important, dont change
        b_name = '_'.join([self._name, 'b'])  # Name is important, dont change

        w = tf.get_variable(
            w_name,
            [self._kernel_size, self._input_dims],
            trainable=False,  # Important: this is constant!
            collections=[KERNEL_COLLECTION, tf.GraphKeys.GLOBAL_VARIABLES]
        )

        b = tf.get_variable(
            b_name,
            [self._kernel_size],
            trainable=False,  # Important: this is constant!
            collections=[KERNEL_COLLECTION, tf.GraphKeys.GLOBAL_VARIABLES]
        )

        w_value, b_value = self.draw_w(), self.draw_b()

        tf.add_to_collection(KERNEL_ASSIGN_OPS, w.assign(w_value))
        tf.add_to_collection(KERNEL_ASSIGN_OPS, b.assign(b_value))

        # Let's store the RFF so we have it if needed
        self._w = w
        self._b = b

        tf.summary.histogram(w_name, w, [tag])
        tf.summary.histogram(b_name, b, [tag])

        # Tensordot sets shape to unknown but the shape of the output is known
        dot = tf.add(tf.tensordot(x, tf.transpose(w), axes=1), b)
        dot.set_shape(x.get_shape().as_list()[:-1] + [self._kernel_size])
        tf.summary.histogram(self._name + '_dot', dot, [tag])

        z = tf.cos(dot) * np.sqrt(2/self._kernel_size)
        tf.summary.histogram(self._name + '_z', z, [tag])

        return z


class GaussianRFF(RandomFourierFeatures):

    def __init__(self,
                 name,
                 input_dims,
                 kernel_size,
                 kernel_std,
                 **params):
        super(GaussianRFF, self).__init__(
            name, input_dims, kernel_size
        )
        self._std = kernel_std

    def draw_w(self):
        return tf.random_normal(
            stddev=self._std,
            shape=[self._kernel_size, self._input_dims]
        )

    def draw_b(self):
        return tf.random_uniform(
            shape=[self._kernel_size], minval=0, maxval=2*np.pi
        )


def is_w(name):
    suffix = name.split(':')[0].split('_')[-1]
    if suffix == 'w':
        return True
    elif suffix == 'b':
        return False
    else:
        raise ValueError('Unexpected variable name')


def sample_w(kernel_fn, var, **params):
    _, input_dims = var.get_shape().as_list()
    kernel = kernel_fn(
        '', input_dims=input_dims, **params
    )
    return kernel.draw_w()


def sample_b(kernel_fn, var, **params):
    input_dims = 128  # This is arbitrary
    kernel = kernel_fn(
        '', input_dims=input_dims, **params
    )
    return kernel.draw_b()


def _generate_w_mask(x, keep_ratio=0.50):
    """
    Returns two masks:
        - A binary mask indicating the values to keep from the input matrix
        - Inverse of the previous mask
    """
    x_shape = x.get_shape().as_list()
    height, width = x_shape

    # Draw numbers in [0,1] and check if x < keep_ratio
    rands = tf.random_uniform(shape=[height], dtype=tf.float32)
    to_keep = tf.less(rands, tf.ones(tf.shape(rands)) * keep_ratio)
    to_keep = tf.cast(to_keep, tf.int32)

    # Get mask so we can use it to replace only the selected rows
    lengths = to_keep * width
    mask = tf.sequence_mask(lengths, width, dtype=tf.float32)

    return mask, tf.subtract(1.0, mask)


def _generate_b_mask(x, keep_ratio=0.50):
    """
    Returns two masks:
        - A binary mask indicating the values to keep from the input vector
        - Inverse of the previous mask
    """
    length = x.get_shape().as_list()[0]

    # Draw numbers in [0,1] and check if x < keep_ratio
    rands = tf.random_uniform(shape=[length], dtype=tf.float32)
    to_keep = tf.less(rands, tf.ones(tf.shape(rands)) * keep_ratio)
    mask = tf.cast(to_keep, tf.float32)

    return mask, tf.subtract(1.0, mask)


def kernel_dropout_w(var, new_sample, keep_ratio):
    """
    Returns the matrix resulting from replacing some rows from the
    new sample according to the given probability ratio
    """
    mask, mask_inv = _generate_w_mask(var, keep_ratio)
    kept = tf.multiply(mask, var)
    sampled = tf.multiply(mask_inv, new_sample)
    return tf.add(kept, sampled)


def kernel_dropout_b(var, new_sample, keep_ratio):
    """
    Returns the vector resulting from replacing some elements
    from the new sample according to the given probability ratio
    """
    mask, mask_inv = _generate_b_mask(var, keep_ratio)
    kept = tf.multiply(mask, var)
    sampled = tf.multiply(mask_inv, new_sample)
    return tf.add(kept, sampled)

