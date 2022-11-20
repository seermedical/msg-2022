# coding=utf-8
from typing import Callable
import tensorflow as tf
import numpy as np


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def _mix_up(
    images_one, labels_one, images_two, labels_two, alpha
):

    batch_size = tf.shape(images_one)[0]
    n_shapes = tf.shape(images_one).shape[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)


    x_l = tf.reshape(l, [batch_size] + [1] * (n_shapes - 1))

    y_l = tf.reshape(l, (batch_size,))
    if tf.shape(labels_one).shape[0] > 1:
        y_l = tf.expand_dims(y_l, axis=-1)
        y_l = tf.repeat(y_l, axis=-1, repeats=tf.shape(labels_one)[-1])

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    x_l = tf.cast(x_l, dtype=images_one.dtype)
    y_l = tf.cast(y_l, dtype=labels_one.dtype)

    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)

    return images, labels

@tf.function
def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    rand = tf.random.uniform((1,), 0, 1)[0]
    if tf.greater(rand, tf.constant(0.5)):
        return _mix_up(
            images_one, labels_one, images_two, labels_two, alpha
        )
    else:
        return images_one, labels_one


def create_mixup_ds(train_ds_one, train_ds_two,):
    train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

    train_data = train_ds.map(
        lambda ds_one, ds_two: mix_up(
            ds_one, ds_two, alpha=0.2
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return train_data
