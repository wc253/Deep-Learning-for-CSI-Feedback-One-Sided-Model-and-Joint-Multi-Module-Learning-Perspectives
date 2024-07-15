from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation, PReLU, \
    GlobalAveragePooling2D, Concatenate, Multiply, Conv2DTranspose, Layer, Add, Permute, RepeatVector, Dropout, \
    AveragePooling2D, subtract, Input
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L1, L2, L1L2


##################################################################################
# FFDNet
##################################################################################
def block_ffdnet(y, norm_style='', dropout=0, weight=0.01, dim=96):
    if norm_style == 'l1':
        y = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L1(weight))(y)
    elif norm_style == 'l2':
        y = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L2(weight))(y)
    elif norm_style == 'l1_l2':
        y = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L1L2(l1=weight, l2=weight))(y)
    else:
        y = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    if dropout > 0:
        y = Dropout(dropout)(y)
    return y


def channel_down(input_tensor, noiseLevel, upscale_factor=2):
    (batch_size, in_height, in_width, channels) = input_tensor.shape
    out_height = int(in_height / upscale_factor)
    out_width = int(in_width / upscale_factor)
    input_reshape = Reshape((out_height, upscale_factor, out_width, upscale_factor, channels), )(input_tensor)
    channels *= upscale_factor ** 2
    input_reshape = Permute((2, 4, 1, 3, 5))(input_reshape)
    input_reshape = Reshape((out_height, out_width, channels))(input_reshape)
    noiseLevel = RepeatVector(out_width * out_height)(noiseLevel)
    noiseLevel = Reshape((out_height, out_width, 1))(noiseLevel)
    input_cat = Concatenate()([input_reshape, noiseLevel])
    return input_cat


def channel_model(input_tensor, block_num=12, upscale_factor=2, norm_style='', dropout=0, weight=0.01, dim=96):
    if norm_style == 'l1':
        x = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L1(weight))(input_tensor)
    elif norm_style == 'l2':
        x = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L2(weight))(input_tensor)
    elif norm_style == 'l1_l2':
        x = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L1L2(l1=weight, l2=weight))(input_tensor)
    else:
        x = Conv2D(dim, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal')(input_tensor)
    x = Activation('relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)

    for i in range(block_num):
        x = block_ffdnet(x, norm_style=norm_style, dropout=dropout, weight=weight, dim=dim)

    if norm_style == 'l1':
        x = Conv2D(2 * upscale_factor * upscale_factor, kernel_size=(3, 3), padding='same',
                   kernel_initializer='truncated_normal', kernel_regularizer=L1(weight))(x)
    elif norm_style == 'l2':
        x = Conv2D(2 * upscale_factor * upscale_factor, kernel_size=(3, 3), padding='same',
                   kernel_initializer='truncated_normal', kernel_regularizer=L2(weight))(x)
    elif norm_style == 'l1_l2':
        x = Conv2D(2 * upscale_factor * upscale_factor, kernel_size=(3, 3), padding='same',
                   kernel_initializer='truncated_normal', kernel_regularizer=L1L2(l1=weight, l2=weight))(x)
    else:
        x = Conv2D(2 * upscale_factor * upscale_factor, kernel_size=(3, 3), padding='same',
                   kernel_initializer='truncated_normal')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    x = Activation('tanh')(x)

    return x


def channel_up(input_tensor, upscale_factor=2):
    (batch_size, in_height, in_width, channels) = input_tensor.shape
    channels /= upscale_factor ** 2
    channels = int(channels)
    input_reshape = Reshape((upscale_factor, upscale_factor, in_height, in_width, channels))(input_tensor)
    input_reshape = Permute((3, 1, 4, 2, 5))(input_reshape)
    out_height = int(in_height * upscale_factor)
    out_width = int(in_width * upscale_factor)
    input_reshape = Reshape((out_height, out_width, channels))(input_reshape)
    return input_reshape


def fddnet(channel_input, channel_noise, block_num, norm, dropout, weight, dim):
    channel_connect = channel_down(channel_input, channel_noise)
    channel_fea = channel_model(channel_connect, block_num=block_num, norm_style=norm, dropout=dropout, weight=weight,
                                dim=dim)
    channel_output = channel_up(channel_fea)
    return channel_output


##################################################################################
# CBDNet
##################################################################################
def block_cbdnet(y, norm_style='', dropout=0, weight=0.01, output_shape=32):
    if norm_style == 'l1':
        y = Conv2D(output_shape, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L1(weight))(y)
    elif norm_style == 'l2':
        y = Conv2D(output_shape, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L2(weight))(y)
    elif norm_style == 'l1_l2':
        y = Conv2D(output_shape, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L1L2(l1=weight, l2=weight))(y)
    else:
        y = Conv2D(output_shape, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal')(y)
    y = Activation('relu')(y)
    if dropout > 0:
        y = Dropout(dropout)(y)
    return y


def out_block_cbdnet(y, norm_style='', dropout=0, weight=0.01, output_shape=32):
    if norm_style == 'l1':
        y = Conv2D(output_shape, kernel_size=(1, 1), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L1(weight))(y)
    elif norm_style == 'l2':
        y = Conv2D(output_shape, kernel_size=(1, 1), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L2(weight))(y)
    elif norm_style == 'l1_l2':
        y = Conv2D(output_shape, kernel_size=(1, 1), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L1L2(l1=weight, l2=weight))(y)
    else:
        y = Conv2D(output_shape, kernel_size=(1, 1), padding='same', kernel_initializer='truncated_normal')(y)
    y = Activation('relu')(y)
    if dropout > 0:
        y = Dropout(dropout)(y)
    return y


def up_cbdnet(x1, x2, output_channels):
    deconv = Conv2DTranspose(output_channels, (2, 2), strides=(2, 2), padding='same')(x1)
    deconv_output = deconv + x2
    deconv_output.set_shape([None, None, None, output_channels])
    return deconv_output


def DnnE(input_tensor, block_num=12, norm_style='', dropout=0, weight=0.01):
    for i in range(block_num):
        x = block_cbdnet(input_tensor, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=32)

    if norm_style == 'l1':
        x = Conv2D(2, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L1(weight))(x)
    elif norm_style == 'l2':
        x = Conv2D(2, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L2(weight))(input_tensor)
    elif norm_style == 'l1_l2':
        x = Conv2D(2, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal',
                   kernel_regularizer=L1L2(l1=weight, l2=weight))(input_tensor)
    else:
        x = Conv2D(2, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal')(input_tensor)
    x = Activation('relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    return x


def DnnD(input_tensor, norm_style='', dropout=0, weight=0.01):
    conv1 = block_cbdnet(input_tensor, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=64)
    conv1 = block_cbdnet(conv1, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=64)
    pool1 = AveragePooling2D(pool_size=[2, 2], padding='SAME')(conv1)

    conv2 = block_cbdnet(pool1, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=128)
    conv2 = block_cbdnet(conv2, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=128)
    conv2 = block_cbdnet(conv2, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=128)
    pool2 = AveragePooling2D(pool_size=[2, 2], padding='SAME')(conv2)

    conv3 = block_cbdnet(pool2, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=256)
    conv3 = block_cbdnet(conv3, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=256)
    conv3 = block_cbdnet(conv3, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=256)
    conv3 = block_cbdnet(conv3, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=256)
    conv3 = block_cbdnet(conv3, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=256)
    conv3 = block_cbdnet(conv3, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=256)
    up3 = up_cbdnet(conv3, conv2, output_channels=128)

    conv4 = block_cbdnet(up3, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=128)
    conv4 = block_cbdnet(conv4, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=128)
    conv4 = block_cbdnet(conv4, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=128)
    up4 = up_cbdnet(conv4, conv1, output_channels=64)

    conv5 = block_cbdnet(up4, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=64)
    conv5 = block_cbdnet(conv5, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=64)

    out = out_block_cbdnet(conv5, norm_style=norm_style, dropout=dropout, weight=weight, output_shape=2)

    return out


def cbdnet(input_tensor, block_num, norm, dropout, weight):
    noise_map = DnnE(input_tensor, block_num=block_num, norm_style=norm, dropout=dropout, weight=weight)
    concat_map = tf.concat([input_tensor, noise_map], 3)
    out = DnnD(concat_map, norm_style=norm, dropout=dropout, weight=weight) + input_tensor
    return noise_map, out


##################################################################################
# ChannelNet
##################################################################################
def DNCNN_model(input_tensor):
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    x = Activation('relu')(x)
    for i in range(8):
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
    x = Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = subtract([input_tensor, x])  # input - noise
    return x


def SRCNN_model(input_tensor):
    x_p = tf.keras.layers.ZeroPadding2D(4)(input_tensor)
    c1 = Conv2D(16, 9, strides=1, activation='relu', kernel_initializer='random_uniform')(x_p)
    c2 = Conv2D(8, 1, strides=1, activation='relu', kernel_initializer='random_uniform')(c1)
    c3_p = tf.keras.layers.ZeroPadding2D(2)(c2)
    c3 = Conv2D(2, 5, strides=1, kernel_initializer='random_uniform')(c3_p)
    return c3


def channelnet(channel_input):
    channel_sr = SRCNN_model(channel_input)
    channel_dn = DNCNN_model(channel_sr)
    return channel_dn


##################################################################################
# ReEsNet
##################################################################################
def reesnet(channel_input, Number_of_pilot):
    x_ = Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='same')(channel_input)
    for i in range(4):
        x = Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='same')(x_)
        x = Activation('relu')(x)
        x = Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x_ = x + x_
    x = Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='same')(x_)
    if Number_of_pilot == 256:
        x = Conv2DTranspose(filters=4, kernel_size=(249, 1), strides=(1, 1), padding="valid", output_padding=None, )(x)
    elif Number_of_pilot == 512:
        x = Conv2DTranspose(filters=4, kernel_size=(241, 1), strides=(1, 1), padding="valid", output_padding=None, )(x)
    elif Number_of_pilot == 128:
        x = Conv2DTranspose(filters=4, kernel_size=(253, 1), strides=(1, 1), padding="valid", output_padding=None, )(x)
    channel_output = Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    return channel_output


##################################################################################
# ADEN
##################################################################################
def aden(channel_input):
    (batch_size, in_height, in_width, channels) = channel_input.shape
    number_in = int(in_height * in_width * channels)
    out_dim = 2048
    input_reshape = Reshape((number_in,))(channel_input)

    k_0 = Dense(out_dim)(input_reshape)
    k_0 = Activation('relu')(k_0)

    k_1 = Dense(out_dim)(k_0)
    k_1 = Activation('relu')(k_1)

    k_2_ = Dense(out_dim)(k_1)
    k_2_ = Activation('relu')(k_2_)
    k_2 = k_2_ + k_0

    k_3_ = Dense(out_dim)(k_2)
    k_3_ = Activation('relu')(k_3_)
    k_3 = k_3_ + k_0 + k_1

    channel_output_ = Dense(out_dim)(k_3)
    channel_output_1 = channel_output_ + k_0 + k_1 + k_2
    channel_output = Dense(16384)(channel_output_1)
    output_reshape = Reshape((256, 32, 2))(channel_output)
    return output_reshape


##################################################################################
# CNN
##################################################################################
def cnn(channel_input, Number_of_antenna):
    x_ = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(channel_input)
    x = Activation('relu')(x_)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x_)
    x = Activation('relu')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    if Number_of_antenna == 8:
        x = Conv2DTranspose(filters=16, kernel_size=(1, 25), strides=(1, 1), padding="valid", output_padding=None, )(x)
    elif Number_of_antenna == 16:
        x = Conv2DTranspose(filters=16, kernel_size=(1, 17), strides=(1, 1), padding="valid", output_padding=None, )(x)
    channel_output = Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    return channel_output


if __name__ == '__main__':
    # channel_input = Input(shape=(256, 32, 2))
    # channel_output = channelnet(channel_input)
    # model = Model(inputs=channel_input, outputs=channel_output, name='denoising')
    # model.summary()

    channel_input = Input(shape=(4, 32, 2))
    channel_output = reesnet(channel_input, 128)
    model = Model(inputs=channel_input, outputs=channel_output, name='denoising')
    model.summary()

    # channel_input = Input(shape=(256, 16, 2))
    # channel_output = aden(channel_input)
    # model = Model(inputs=channel_input, outputs=channel_output, name='denoising')
    # model.summary()

    # channel_input = Input(shape=(256, 16, 2))
    # channel_output = cnn(channel_input, 16)
    # model = Model(inputs=channel_input, outputs=channel_output, name='denoising')
    # model.summary()
