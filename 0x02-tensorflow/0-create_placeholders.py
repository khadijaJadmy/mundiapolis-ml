#!/usr/bin/env python3

import tensorflow as tf


def create_placeholders(nx, classes):
   """
   """
    placeholder1=tf.placeholder(float, shape=[None, nx], name='x')
    placeholder2= tf.placeholder(float, shape=[None, classes], name='y')
    return placeholder1,placeholder2
