from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation, PReLU, \
    GlobalAveragePooling2D, Concatenate, Multiply, Conv2DTranspose, Layer, Add
import tensorflow as tf


def add_common_layers(y):
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y


def CsiNetPlusEncoder(input_tensor, encoded_dim):
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(input_tensor)
    x = add_common_layers(x)
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(x)
    x = add_common_layers(x)
    x = Reshape((32 * 32 * 2,))(x)
    encoded = Dense(encoded_dim, activation='tanh', kernel_initializer='truncated_normal', name='encoded')(x)
    return encoded


def CsiNetPlusDecoder(input_tensor, residual_num=5):
    def residual_block_decoded(y):
        shortcut = y
        y = Conv2D(8, kernel_size=(7, 7), padding='same', kernel_initializer='truncated_normal')(y)
        y = add_common_layers(y)

        y = Conv2D(16, kernel_size=(5, 5), padding='same', kernel_initializer='truncated_normal')(y)
        y = add_common_layers(y)

        y = Conv2D(2, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal')(y)
        y = BatchNormalization()(y)
        y = Activation('tanh')(y)

        y = add([shortcut, y])

        return y

    x = Dense(32 * 32 * 2, activation='linear', kernel_initializer='truncated_normal', name='decoded')(input_tensor)
    x = Reshape((32, 32, 2))(x)
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    for i in range(residual_num):
        x = residual_block_decoded(x)

    x = Activation('relu')(x)
    return x


def TransEncoder(x):
    x = Conv2D(16, (3, 3), strides=(2, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(16, (3, 3), strides=(2, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2D(2, (3, 3), strides=(2, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    return x


def TransDecoder(x):
    x = Conv2DTranspose(16, (3, 3), strides=(2, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2DTranspose(16, (3, 3), strides=(2, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv2DTranspose(2, (3, 3), strides=(2, 1), padding='same')(x)
    x = Activation('sigmoid')(x)
    return x


class MuCompandLayer(Layer):
    def __init__(self, mu=50., **kwargs):
        self.mu = mu
        super(MuCompandLayer, self).__init__(**kwargs)

    def call(self, x):
        sign_x = tf.greater(x, 0)
        sign_x = 2 * (tf.cast(sign_x, dtype=tf.float32) - 0.5)
        out = sign_x * tf.math.log(1 + self.mu * sign_x * x) / tf.math.log(1 + self.mu)
        return out


class DeMuCompandLayer(Layer):
    def __init__(self, mu=50., **kwargs):
        self.mu = mu
        super(DeMuCompandLayer, self).__init__(**kwargs)

    def call(self, x):
        sign_x = tf.greater(x, 0)
        sign_x = 2 * (tf.cast(sign_x, dtype=tf.float32) - 0.5)
        out = sign_x * ((1 + self.mu) ** (sign_x * x) - 1) / self.mu
        return out


@tf.custom_gradient
def quan_value(x, B):
    def grad(dy):
        return dy, dy

    # Due to the use of tanh in the output of the encoder, x should be converted from [-1,1] to [0,1]
    x = tf.divide(tf.add(x, 1), 2)
    level = tf.cast(2 ** B, dtype=tf.float32)
    q_value = tf.round(x * level - 0.5)
    return q_value, grad


class QuantizationLayer(Layer):
    def __init__(self, B, **kwargs):
        self.B = B
        super(QuantizationLayer, self).__init__(**kwargs)

    def call(self, x):
        return quan_value(x, self.B)


@tf.custom_gradient
def dequan_value(x, B):
    def grad(dy):
        return dy, dy

    x = tf.cast(x, dtype=tf.float32)
    level = tf.cast(2 ** B, dtype=tf.float32)
    dq_value = (x + 0.5) / level
    # dq_value from [0,1] to [-1,1]
    dq_value = dq_value * 2 - 1
    return dq_value, grad


class DequantizationLayer(Layer):
    def __init__(self, B, **kwargs):
        self.B = B
        super(DequantizationLayer, self).__init__(**kwargs)

    def call(self, x):
        return dequan_value(x, self.B)


def Offset(x, encoded_dim):
    short_cut = x
    for i in range(3):
        x = Dense(encoded_dim, )(x)
        x = LeakyReLU()(x)
    x = add([short_cut, x])
    return x


def QuanDequanOffset(input_tensor, B, encoded_dim):
    x = MuCompandLayer()(input_tensor)
    x = QuantizationLayer(B=B)(x)
    x = DequantizationLayer(B=B)(x)
    x = DeMuCompandLayer()(x)
    x = Offset(x, encoded_dim)
    return x


def AF_Module(inputs, snr):
    (_, width, height, ch_num) = inputs.shape
    m = GlobalAveragePooling2D()(inputs)
    m = Concatenate()([m, snr])
    m = Dense(16, activation='relu')(m)
    m = Dense(ch_num, activation='sigmoid')(m)
    out = Multiply()([inputs, m])
    return out


def AFTransEncoder(x, snr):
    x = Conv2D(16, (3, 3), strides=(2, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = AF_Module(x, snr)
    x = Conv2D(16, (3, 3), strides=(2, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = AF_Module(x, snr)
    x = Conv2D(2, (3, 3), strides=(2, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    return x


def AFTransDecoder(x, snr):
    x = Conv2DTranspose(16, (3, 3), strides=(2, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = AF_Module(x, snr)
    x = Conv2DTranspose(16, (3, 3), strides=(2, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = AF_Module(x, snr)
    x = Conv2DTranspose(2, (3, 3), strides=(2, 1), padding='same')(x)
    x = Activation('sigmoid')(x)
    return x


def AFCsiNetPlusEncoder(input_tensor, encoded_dim, snr):
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(input_tensor)
    x = add_common_layers(x)
    x = AF_Module(x, snr)
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(x)
    x = add_common_layers(x)
    x = AF_Module(x, snr)
    x = Reshape((32 * 32 * 2,))(x)
    encoded = Dense(encoded_dim, activation='tanh', kernel_initializer='truncated_normal', name='encoded')(x)
    return encoded


def AFCsiNetPlusDecoder(input_tensor, snr, residual_num=5):
    def residual_block_decoded(y, snr):
        shortcut = y
        y = Conv2D(8, kernel_size=(7, 7), padding='same', kernel_initializer='truncated_normal')(y)
        y = add_common_layers(y)
        y = AF_Module(y, snr)

        y = Conv2D(16, kernel_size=(5, 5), padding='same', kernel_initializer='truncated_normal')(y)
        y = add_common_layers(y)
        y = AF_Module(y, snr)

        y = Conv2D(2, kernel_size=(3, 3), padding='same', kernel_initializer='truncated_normal')(y)
        y = BatchNormalization()(y)
        y = Activation('tanh')(y)
        y = AF_Module(y, snr)

        y = add([shortcut, y])

        return y

    x = Dense(32 * 32 * 2, activation='linear', kernel_initializer='truncated_normal', name='decoded')(input_tensor)
    x = Reshape((32, 32, 2))(x)
    x = AF_Module(x, snr)
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = AF_Module(x, snr)

    for i in range(residual_num):
        x = residual_block_decoded(x, snr)

    x = Activation('relu')(x)
    return x
