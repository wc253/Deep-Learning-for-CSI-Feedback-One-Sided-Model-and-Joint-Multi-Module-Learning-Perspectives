import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Flatten


class UplinkChannelMRC(Layer):
    def __init__(self, power_per_symbol=1, **kwargs):
        self.power_per_symbol = power_per_symbol
        super(UplinkChannelMRC, self).__init__(**kwargs)

    def call(self, features, h_real, snr_db):
        inter_shape = tf.shape(features)
        # convert to complex channel signal
        f = Flatten()(features)
        dim_z = tf.shape(f)[1] // 2  # dim_z == 32/2 == 16
        z_in = tf.complex(f[:, :dim_z], f[:, dim_z:])
        # power constraint, the average complex symbol power is 1
        norm_factor = tf.reduce_sum(
            tf.math.real(z_in * tf.math.conj(z_in)), axis=1, keepdims=True
        )
        #  z_in_norm = z_in * sqrt(16/(z_in*z_in‘))  使得平均功率归一化
        z_in_norm = z_in * tf.complex(
            tf.sqrt(tf.cast(dim_z, dtype=tf.float32) / norm_factor), 0.0
        )
        # add fading
        h = tf.complex(h_real[:, :, :, 0], h_real[:, :, :, 1])
        z_in_norm = z_in_norm[..., tf.newaxis]  # 增加维度
        z_in_transmit = np.sqrt(self.power_per_symbol) * z_in_norm  # 发送信号
        fd = tf.multiply(z_in_transmit, h)
        # add noise
        rv = self.add_noise(fd, snr_db)
        '''
        MRC:
            Y = Hx+n
            WH*Y = WH*(Hx+n)
            WH is Hermitian of W, W=H/H的2范数
        '''
        # 在天线维度计算
        h_norm2 = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(h)), axis=2, keepdims=True))
        h_norm2_c = tf.complex(h_norm2, 0 * h_norm2)
        w = h / h_norm2_c
        mrc = tf.reduce_sum(tf.multiply(rv, tf.math.conj(w)), axis=2)
        z2r = tf.concat([tf.math.real(mrc), tf.math.imag(mrc)], 1)
        out = tf.reshape(z2r, inter_shape)
        return out

    def add_noise(self, fd, snr_db):
        # 对接收信号计算接收功率，计算接收信噪比
        # signal_power = tf.reduce_mean(tf.abs(fd)**2, axis=(1,2), keepdims=True)
        # noise_power = signal_power/(10 ** (snrdb / 10))
        # 将信道归一化后
        noise_std = tf.sqrt(10 ** (-snr_db / 10))
        noise_std = tf.complex(noise_std, 0.)
        noise_std = tf.reshape(noise_std, [-1, 1, 1])
        noise_normal = tf.complex(tf.random.normal(tf.shape(fd), 0, 1 / np.sqrt(2)),
                                  tf.random.normal(tf.shape(fd), 0, 1 / np.sqrt(2)))
        return fd + noise_std * noise_normal


