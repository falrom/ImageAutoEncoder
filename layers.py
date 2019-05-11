import tensorflow as tf
import numpy as np
from utils import cv_imread
import matplotlib.pyplot as plt
import os


def deconv_subpixel(inputs, factor, name, add_front_conv=False, activation=None):
    inputs = tf.convert_to_tensor(inputs)
    bi, hi, wi, ci = inputs.shape.as_list()
    num = factor ** 2

    with tf.variable_scope(name_or_scope=name):

        # front convolution layer:
        if add_front_conv:
            ci = ci * num
            inputs = tf.layers.conv2d(inputs, filters=ci, kernel_size=3, strides=(1, 1), padding='same',
                                      name='front_conv')

        # build filters:
        assert ci % num == 0
        co = ci // num
        filters = np.zeros([factor, factor, co, ci], np.float32)
        for o in range(co):
            for i in range(num):
                filters[i // factor, i % factor, o, o * num + i] = 1

        # deconv:
        outputs = tf.layers.conv2d_transpose(inputs, filters=co, kernel_size=factor, strides=(factor, factor),
                                             padding='same', activation=activation, use_bias=False,
                                             kernel_initializer=tf.constant_initializer(filters), trainable=False,
                                             name='subpix_deconv')
        # outputs = tf.nn.conv2d_transpose(inputs, filter=filters, output_shape=(bi, hi * factor, wi * factor, co),
        #                                  strides=(1, factor, factor, 1), padding='SAME', name='subpix_deconv')

    return outputs


def _test_conv_subpixel():
    factor = 3
    inputs = [[], [], []]
    for f in range(factor ** 2):
        img = (cv_imread('tests/deconv_subpixel/' + str(f + 1) + '.png', 'RGB') / 255).astype(np.float32)
        plt.figure()
        plt.imshow(img)
        plt.title(str(f + 1))
        for c in range(3):
            inputs[c].append(img[np.newaxis, :, :, c:c + 1])
    for c in range(3):
        inputs[c] = np.concatenate(inputs[c], 3)
    inputs = np.concatenate(inputs, 3)
    outputs = deconv_subpixel(inputs, factor, 'conv_sp')
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    img_o = sess.run(outputs)
    plt.figure()
    plt.imshow(img_o[0, :])
    plt.title('output')
    plt.show()
    sess.close()
    os.system('PAUSE')


def conv_autoencoder(inputs, ratio, channels, name, use_pooling=False, use_subpixel=True):
    factor = 2
    inputs = tf.convert_to_tensor(inputs)
    bi, hi, wi, ci = inputs.shape.as_list()

    with tf.variable_scope(name):

        with tf.variable_scope('Encoder'):  # encoder:
            conv_bgn = tf.layers.conv2d(inputs, filters=channels, kernel_size=3, strides=(1, 1), padding='same',
                                        activation=tf.nn.relu, name='conv_bgn')
            # conv_bgn_2 = tf.layers.conv2d(conv_bgn_1, filters=channels, kernel_size=3, strides=(1, 1), padding='same',
            #                               activation=tf.nn.relu, name='conv_bgn_2')
            node = conv_bgn

            for r in range(ratio):
                if use_pooling:
                    node = tf.layers.max_pooling2d(node, 2, (factor, factor), padding='same', name='pooling_' + str(r))
                    strides_2 = (1, 1)
                else:
                    strides_2 = (factor, factor)
                node = tf.layers.conv2d(node, filters=channels, kernel_size=3, strides=strides_2, padding='same',
                                        activation=tf.nn.relu, name='conv_' + str(r) + '_1')
                node = tf.layers.conv2d(node, filters=channels, kernel_size=3, strides=(1, 1), padding='same',
                                        activation=tf.nn.relu, name='conv_' + str(r) + '_2')
        encoded = node

        with tf.variable_scope('Decoder'):  # decoder:
            for r in range(ratio - 1, 0 - 1, -1):
                node = tf.layers.conv2d(node, filters=channels, kernel_size=3, strides=(1, 1), padding='same',
                                        activation=tf.nn.relu, name='conv_' + str(r) + '_1')
                if use_subpixel:
                    node = deconv_subpixel(node, factor=factor, name='subpix_' + str(r), add_front_conv=True,
                                           activation=tf.nn.relu)
                else:
                    node = tf.layers.conv2d_transpose(node, channels, 3, (factor, factor), padding='same',
                                                      activation=tf.nn.relu, name='deconv_' + str(r))

            conv_end = tf.layers.conv2d(node, filters=ci, kernel_size=3, strides=(1, 1), padding='same',
                                        name='conv_end')
        decoded = conv_end

    return encoded, decoded


def _get_conv_autoencoder_graph(log_path='tests/conv_autoencoder/log'):
    log_writer = tf.summary.FileWriter(log_path)
    inputs = tf.placeholder(tf.float32, [None, 64, 64, 3], 'inputs')
    conv_autoencoder(inputs, 0, 32, 'AE')
    log_writer.add_graph(tf.get_default_graph())
    log_writer.flush()


def MSE_float(predictions, labels):
    return tf.losses.mean_squared_error(labels=labels, predictions=predictions, scope='MSE_float')


def loss_L1(predictions, labels):
    return tf.losses.absolute_difference(labels=labels, predictions=predictions, scope='L1_loss')


def PSNR_float(mse):
    with tf.name_scope(name='PSNR_float'):
        psnr = tf.multiply(10.0, tf.log(1.0 * 1.0 / mse) / tf.log(10.0))
    return psnr


def Outputs(predictions):
    with tf.name_scope(name='Outputs'):
        results = tf.clip_by_value(tf.round(255.0 * predictions), 0, 255)
        results = tf.cast(results, tf.uint8)
    return results


def PSNR_uint8(outputs, labels):
    with tf.name_scope(name='PSNR_uint8'):
        with tf.name_scope(name='MSE'):
            mse = tf.reduce_mean(
                tf.square(tf.clip_by_value(tf.round(255.0 * labels), 0, 255) - tf.cast(outputs, tf.float32)))
        psnr = tf.multiply(10.0, tf.log(255.0 * 255.0 / mse) / tf.log(10.0))
    return psnr


if __name__ == '__main__':
    _get_conv_autoencoder_graph()
