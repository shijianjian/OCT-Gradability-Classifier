"""A vanilla 3D resnet implementation.

Based on Raghavendra Kotikalapudi's 2D implementation
keras-resnet (See https://github.com/raghakot/keras-resnet.)
"""
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)
from math import ceil
from tensorflow.keras.models import (
    Model
)
from tensorflow.keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from tensorflow.keras.layers import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D,
    BatchNormalization,
    add,
    GlobalAveragePooling3D,
    Reshape,
    Multiply
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import tensorflow as tf


def _bn_relu(input):
    """Helper to build a BN -> relu block """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu3D(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault(
        "kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))

    def f(input):
        conv = Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv3d(**conv_params):
    """Helper to build a  BN -> relu -> conv3d block."""
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(activation)
    return f


def _shortcut3d(input, residual):
    """3D shortcut to match input and residual and merges them with "sum"."""
    stride_dim1 = ceil((input.get_shape().as_list()[DIM1_AXIS] / residual.get_shape().as_list()[DIM1_AXIS]))
    stride_dim2 = ceil((input.get_shape().as_list()[DIM2_AXIS] / residual.get_shape().as_list()[DIM2_AXIS]))
    stride_dim3 = ceil((input.get_shape().as_list()[DIM3_AXIS] / residual.get_shape().as_list()[DIM3_AXIS]))
    equal_channels = residual.get_shape().as_list()[CHANNEL_AXIS] == input.get_shape().as_list()[CHANNEL_AXIS]

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
        shortcut = Conv3D(
            filters=residual.get_shape().as_list()[CHANNEL_AXIS],
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            kernel_initializer="he_normal", padding="valid",
            kernel_regularizer=l2(1e-4)
        )(input)
    return add([shortcut, residual])


def _residual_block3d(block_function, filters, kernel_regularizer, repetitions,
                      is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            strides = (1, 1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2, 2)
            input = block_function(filters=filters, strides=strides,
                                   kernel_regularizer=kernel_regularizer,
                                   is_first_block_of_first_layer=(
                                       is_first_layer and i == 0)
                                   )(input)
        return input

    return f


def basic_block(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
                is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           strides=strides, padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=kernel_regularizer
                           )(input)
        else:
            conv1 = _bn_relu_conv3d(filters=filters,
                                    kernel_size=(3, 3, 3),
                                    strides=strides,
                                    kernel_regularizer=kernel_regularizer
                                    )(input)

        residual = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv1)
        return _shortcut3d(input, residual)

    return f


def bottleneck(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
               is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
                              strides=strides, padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=kernel_regularizer
                              )(input)
        else:
            conv_1_1 = _bn_relu_conv3d(filters=filters, kernel_size=(1, 1, 1),
                                       strides=strides,
                                       kernel_regularizer=kernel_regularizer
                                       )(input)

        conv_3_3 = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv_1_1)
        residual = _bn_relu_conv3d(filters=filters * 4, kernel_size=(1, 1, 1),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv_3_3)

        return _shortcut3d(input, residual)

    return f


def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4


def squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name):

        squeeze = GlobalAveragePooling3D()(input_x)

        excitation = Dense(units=out_dim / ratio)(squeeze)
        excitation = Activation("relu")(excitation)
        excitation = Dense(units=out_dim)(excitation)
        excitation = Activation("sigmoid")(excitation)
        excitation = Reshape([1, 1, 1, out_dim])(excitation)
        scale = Multiply()([input_x, excitation])

        return scale


class Resnet3DBuilder(object):
    """ResNet3D."""

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, reg_factor, with_SE=False):
        """Instantiate a vanilla ResNet3D keras model.

        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            num_outputs: The number of outputs at the final softmax layer
            block_fn: Unit block to use {'basic_block', 'bottlenack'}
            repetitions: Repetitions of unit blocks
        # Returns
            model: a 3D ResNet model that takes a 5D tensor (volumetric images
            in batch) as input and returns a 1D vector (prediction) as output.
        """
        _handle_data_format()
        if len(input_shape) != 4:
            raise ValueError("Input shape should be a tuple "
                             "(conv_dim1, conv_dim2, conv_dim3, channels) "
                             "for tensorflow as backend or "
                             "(channels, conv_dim1, conv_dim2, conv_dim3) "
                             "for theano as backend")
        if block_fn == 'basic_block':
            block_fn = basic_block
        elif block_fn == 'bottleneck':
            block_fn = bottleneck
        else:
            raise ValueError('Block function `%s` not found' % block_fn)

        input = Input(shape=input_shape)
        # first conv 64 filters
        conv1 = _conv_bn_relu3D(filters=64, kernel_size=(7, 7, 7),
                                strides=(2, 2, 2),
                                kernel_regularizer=l2(reg_factor)
                                )(input)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2),
                             padding="same")(conv1)

        # repeat blocks
        block = pool1
        filters = 64  # shall be 64
        for i, r in enumerate(repetitions):
            block = _residual_block3d(block_fn, filters=filters,
                                      kernel_regularizer=l2(reg_factor),
                                      repetitions=r, is_first_layer=(i == 0)
                                      )(block)
            if with_SE:
                block = squeeze_excitation_layer(block, filters, 4, 'se_%d' % i)
            filters *= 2

        # last activation
        block_output = _bn_relu(block)

        # average poll and classification
        pool2 = AveragePooling3D(pool_size=(block.get_shape().as_list()[DIM1_AXIS],
                                            block.get_shape().as_list()[DIM2_AXIS],
                                            block.get_shape().as_list()[DIM3_AXIS]),
                                 strides=(1, 1, 1))(block_output)
        flatten1 = Flatten()(pool2)
        if num_outputs > 1:
            dense = Dense(units=num_outputs,
                          kernel_initializer="he_normal",
                          activation="softmax",
                          kernel_regularizer=l2(reg_factor))(flatten1)
        else:
            dense = Dense(units=num_outputs,
                          kernel_initializer="he_normal",
                          activation="sigmoid",
                          kernel_regularizer=l2(reg_factor))(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, block_fn="basic_block", reg_factor=1e-4, with_SE=False):
        """Build resnet 18-like with 5 blocks."""
        return Resnet3DBuilder.build(input_shape, num_outputs, block_fn,
                                     [2, 2, 2, 2, 2], reg_factor=reg_factor, with_SE=with_SE)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, block_fn="basic_block", reg_factor=1e-4):
        """Build resnet 34."""
        return Resnet3DBuilder.build(input_shape, num_outputs, block_fn,
                                     [3, 4, 6, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, block_fn="bottleneck", reg_factor=1e-4):
        """Build resnet 50."""
        return Resnet3DBuilder.build(input_shape, num_outputs, block_fn,
                                     [3, 4, 6, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, block_fn="bottleneck", reg_factor=1e-4):
        """Build resnet 101."""
        return Resnet3DBuilder.build(input_shape, num_outputs, block_fn,
                                     [3, 4, 23, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, block_fn="bottleneck", reg_factor=1e-4):
        """Build resnet 152."""
        return Resnet3DBuilder.build(input_shape, num_outputs, block_fn,
                                     [3, 8, 36, 3], reg_factor=reg_factor)


if __name__ == '__main__':

    from tensorflow.python.keras.optimizers import Adam
    import tensorflow as tf
    import numpy as np
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['KERAS_BACKEND'] = 'tensorflow'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.allow_soft_placement = True
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    K.set_session(sess)  # set this TensorFlow session as the default session for Keras

    target_shape = (200, 1024, 200, 1)
    data_size = 10
    num_outputs = 2

    model = Resnet3DBuilder.build_resnet_18(target_shape, num_outputs, block_fn="basic_block")

    adam = Adam(lr=1e-3, amsgrad=True)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['accuracy'])
    print(model.summary())

    # Mimic training
    # Data Preparation
    sample_data = np.random.random((data_size, *target_shape))
    sample_raw_labels = np.random.randint(0, num_outputs, data_size, dtype=int)
    sample_labels = np.zeros((data_size, num_outputs))
    sample_labels[np.arange(data_size), sample_raw_labels] = 1
    # Training
    model.fit(sample_data, sample_labels, epochs=2, batch_size=1)
