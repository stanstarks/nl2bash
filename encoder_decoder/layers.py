"""


Some functions are taken from QANet Library:
https://github.com/NLPLearn/QANet/blob/master/layers.py
"""

import tensorflow as tf
import math


initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                     mode='FAN_AVG',
                                                                     uniform=True,
                                                                     dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                          mode='FAN_IN',
                                                                          uniform=False,
                                                                          dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)


def conv(inputs, output_size, bias=None, activation=None, kernel_size=1, name="conv", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 3:
            raise NotImplementedError
        filter_shape = [kernel_size, shapes[-1], output_size]
        bias_shape = [1, 1, output_size]
        strides = 1
    conv_func = tf.nn.conv1d
    kernel_ = tf.get_variable('kernel_',
                              filter_shape,
                              dtype=tf.float32,
                              regularizer=regularizer,
                              initializer=initializer_relu() if activation is not None else initializer())
    outputs = conv_func(inputs, kernel_, strides, "VALID")
    if bias:
        outputs += tf.get_variable("bias_",
                                   bias_shape,
                                   regularizer=regularizer,
                                   initializer=tf.zero_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal


def layer_dropout(inputs, residual, dropout):
    pred = tf.random_uniform([]) < dropout
    return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)


def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
            "layer_norm_scale", [filters], regularizer=regularizer, initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters], regularizer=regularizer, initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result


norm_fn = layer_norm  # tf.contrib.layers.layer_norm #tf.contrib.layers.layer_norm or noam_norm


def depthwise_separable_convolution(inputs, kernel_size, num_filters,
                                    scope="depthwise_separable_convolution",
                                    bias=True, is_training=True, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shapes = inputs.shape.as_list()
        depthwise_filter = tf.get_variable("depthwise_filter",
                                           (kernel_size[0], kernel_size[1], shapes[-1], 1),
                                           dtype=tf.float32,
                                           regularizer=regularizer,
                                           initializer=initializer_relu())
        pointwise_filter = tf.get_variable("pointwise_filter",
                                           (1, 1, shapes[-1], num_filters),
                                           dtype=tf.float32,
                                           regularizer=regularizer,
                                           initializer=initializer_relu())
        outputs = tf.nn.separable_conv2d(inputs,
                                         depthwise_filter,
                                         pointwise_filter,
                                         strides=(1, 1, 1, 1),
                                         padding="SAME")
        if bias:
            b = tf.get_variable("bias",
                                outputs.shape[-1],
                                regularizer=regularizer,
                                initializer=tf.zeros_initializer())
            outputs += b
        outputs = tf.nn.relu(outputs)
        return outputs


def conv_block(inputs, num_conv_layers, kernel_size, num_filters,
               seq_len=None, scope="conv_block", is_training=True,
               reuse=None, bias=True, dropout=0.0, sublayers=(1, 1)):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.expand_dims(inputs, 2)
        l, L = sublayers
        for i in range(num_conv_layers):
            residual = outputs
            outputs = norm_fn(outputs, scope="layer_norm_%d"%i, reuse=reuse)
            if (i) % 2 == 0:
                outputs = tf.nn.dropout(outputs, 1.0 - dropout)
            outputs = depthwise_separable_convolution(outputs,
                                                      kernel_size=(kernel_size, 1),
                                                      num_filters=num_filters,
                                                      scope="depthwise_conv_layers_%d" % i,
                                                      is_training=is_training, reuse=reuse)
            outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
            l += 1
        return tf.squeeze(outputs, 2), l


def multihead_attention(queries, units, num_heads,
                        memory=None,
                        seq_len=None,
                        scope="Multi_Head_Attention",
                        reuse=None,
                        mask=None,
                        is_training=True,
                        bias=True,
                        dropout=0.0):
    with tf.variable_scope(scope, reuse=reuse):
        # Self attention
        if memory is None:
            memory = queries

        memory = conv(memory, 2 * units, name="memory_projection", reuse=reuse)
        query = conv(queries, units, name="query_projection", reuse=reuse)
        Q = split_last_dimension(query, num_heads)
        K, V = [split_last_dimension(tensor, num_heads) for tensor in tf.split(memory, 2, axis=2)]

        key_depth_per_head = units // num_heads
        Q *= key_depth_per_head**-0.5
        x = dot_product_attention(Q, K, V,
                                  bias=bias,
                                  seq_len=seq_len,
                                  mask=mask,
                                  is_training=is_training,
                                  scope="dot_product_attention",
                                  reuse=reuse, dropout=dropout)
        return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def self_attention_block(inputs, num_filters, seq_len, mask=None, num_heads=8,
                         scope="self_attention_ffn", reuse=None, is_training=True,
                         bias=True, dropout=0.0, sublayers=(1, 1)):
    with tf.variable_scope(scope, reuse=reuse):
        l, L = sublayers
        # Self attention
        outputs = norm_fn(inputs, scope="layer_norm_1", reuse=reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = multihead_attention(outputs, num_filters,
                                      num_heads=num_heads, seq_len=seq_len, reuse=reuse,
                                      mask=mask, is_training=is_training, bias=bias, dropout=dropout)
        residual = layer_dropout(outputs, inputs, dropout * float(l) / L)
        l += 1
        # Feed-forward
        outputs = norm_fn(residual, scope="layer_norm_2", reuse=reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = conv(outputs, num_filters, True, tf.nn.relu, name="FFN_1", reuse=reuse)
        outputs = conv(outputs, num_filters, True, None, name="FFN_2", reuse=reuse)
        outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
        l += 1
        return outputs, l


def residual_block(inputs, num_blocks, num_conv_layers, kernel_size, mask=None,
                   num_filters=128, input_projection=False, num_heads=8,
                   seq_len=None, scope="res_block", is_training=True,
                   reuse=None, bias=True, dropout=0.0):
    with tf.variable_scope(scope, reuse=reuse):
        if input_projection:
            inputs = conv(inputs, num_filters, name="input_projection", reuse=reuse)
        outputs = inputs
        sublayer = 1
        total_sublayers = (num_conv_layers + 2) * num_blocks
        for i in range(num_blocks):
            outputs = add_timing_signal_1d(outputs)
            outputs, sublayer = conv_block(outputs, num_conv_layers, kernel_size, num_filters,
                                           seq_len=seq_len, scope='encoder_block_%d' % i,
                                           reuse=reuse, bias=bias, dropout=dropout,
                                           sublayers=(sublayer, total_sublayers))
            outputs, sublayer = self_attention_block(outputs, num_filters, seq_len, mask=mask,
                                                     num_heads=num_heads,
                                                     scope="self_attention_layer_%d" % i,
                                                     reuse=reuse, is_training=is_training,
                                                     bias=bias, dropout=dropout,
                                                     sublayers=(sublayer, total_sublayers))
        return outputs
