#!/usr/bin/env python3

import tensorflow as tf


def create_layer(previous, n, activation):
  
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                            name="layer")
    return layer(previous)
