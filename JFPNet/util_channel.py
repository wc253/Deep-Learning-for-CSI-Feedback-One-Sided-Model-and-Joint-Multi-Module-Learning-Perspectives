import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Flatten


class UplinkChannelMMSE(Layer):
    def __init__(self, args, **kwargs):
        self.P1 = args.P1
        self.P2 = args.P2
        self.P3 = args.P3
        self.P4 = args.P4
        self.rx = args.rx
        self.tx = args.tx
        self.sample = 200
        super(UplinkChannelMMSE, self).__init__(**kwargs)

    def call(self, tx_1, tx_2, tx_3, tx_4, h_real, snr_dB):
        inter_shape = tf.shape(tx_1)  # 100000*2k

        f1 = Flatten()(tx_1)
        f2 = Flatten()(tx_2)
        f3 = Flatten()(tx_3)
        f4 = Flatten()(tx_4)
        dim_z = tx_1.shape[1] // 2  # 占用子载波数
        s_1 = tf.complex(f1[:, :dim_z], f1[:, dim_z:])  # 100000*k
        s_2 = tf.complex(f2[:, :dim_z], f2[:, dim_z:])  # 100000*k
        s_3 = tf.complex(f3[:, :dim_z], f3[:, dim_z:])  # 100000*k
        s_4 = tf.complex(f4[:, :dim_z], f4[:, dim_z:])  # 100000*k
        # normalized

        s = tf.concat([s_1, s_2, s_3, s_4], axis=1)
        norm_factor = tf.reduce_sum(tf.math.real(s * tf.math.conj(s)), axis=1, keepdims=True)

        s1_norm = tf.expand_dims(s_1 * tf.complex(tf.sqrt(tf.cast(dim_z * 4, dtype=tf.float32) / norm_factor), 0.0),
                                 axis=2)  # 100000*k*1
        s2_norm = tf.expand_dims(s_2 * tf.complex(tf.sqrt(tf.cast(dim_z * 4, dtype=tf.float32) / norm_factor), 0.0),
                                 axis=2)  # 100000*k
        s3_norm = tf.expand_dims(s_3 * tf.complex(tf.sqrt(tf.cast(dim_z * 4, dtype=tf.float32) / norm_factor), 0.0),
                                 axis=2)  # 100000*k
        s4_norm = tf.expand_dims(s_4 * tf.complex(tf.sqrt(tf.cast(dim_z * 4, dtype=tf.float32) / norm_factor), 0.0),
                                 axis=2)  # 100000*k
        tx_1 = np.sqrt(self.P1) * s1_norm
        tx_2 = np.sqrt(self.P2) * s2_norm
        tx_3 = np.sqrt(self.P3) * s3_norm
        tx_4 = np.sqrt(self.P4) * s4_norm
        tx = tf.concat([tx_1, tx_2, tx_3, tx_4], axis=2)  # 100000*k*4
        tx = tf.expand_dims(tx, axis=3)  # 100000*k*4*1

        # add fading
        h = tf.complex(h_real[:, :, :, :, 0], h_real[:, :, :, :, 1])  # 100000*k*32*4
        fd = tf.matmul(h, tx)  # 100000*k*32*1
        # add AWGN
        #norm_power = tf.reduce_mean(tf.square(tf.abs(fd)),axis=[1,2,3])
        #norm_power = tf.reduce_mean(tf.square(tf.abs(fd)), axis=[1, 2, 3], keepdims=True)  # 100000*k*1*1
        #rv = self.add_noise(fd, snr_dB, norm_power)  # 100000*k*32*1
        rv = self.add_noise(fd, snr_dB, 4)
        # MMSE receive
        X_est = self.MMSE_Equ(rv, h, snr_dB)  # 10000*k*4

        z2r = tf.concat([tf.math.real(X_est), tf.math.imag(X_est)], 1)  # 10000*2k*4
        out_1 = z2r[:, :, 0]  # 100000*2k
        out_2 = z2r[:, :, 1]  # 100000*2k
        out_3 = z2r[:, :, 2]  # 100000*2k
        out_4 = z2r[:, :, 3]  # 100000*2k
        return out_1, out_2, out_3, out_4

    def MMSE_Equ(self, y, H, SNR):
        # H:100000*k*32*4
        # y:100000*k*32
        sample = tf.shape(y)[1]
        sigma = 4 * 10 ** (-SNR / 10)
        sigma = tf.complex(sigma, 0.)
        sigma = tf.reshape(sigma, [-1, 1, 1, 1])
        H_H = tf.transpose(H, [0, 1, 3, 2], conjugate=True)  # 100000*4*32
        RHH = tf.matmul(H, H_H)  # 100000*k*32*32
        eye = tf.eye(self.rx, self.rx, dtype=H.dtype)  # 100000*32*32
        inv_Ryy = tf.linalg.inv(tf.math.add(RHH, sigma * eye))  # 100000*k*32*32
        G = tf.matmul(H_H, inv_Ryy)  # 100000*k*4*32
        x_equ = tf.matmul(G, y)  # 100000*k*4*1
        return x_equ
        
        
    def add_noise(self, fd, snr_db, P):
        #noise_std = tf.sqrt(
        #     tf.multiply(P, tf.expand_dims(tf.expand_dims(10 ** (-snr_db / 10), axis=2),axis=3)))  # 100000*1*1*1
        noise_std = tf.sqrt(P * 10 ** (-snr_db / 10))
        noise_std = tf.complex(noise_std, 0.)
        noise_std = tf.reshape(noise_std, [-1, 1, 1, 1])
        noise_normal = tf.complex(tf.random.normal(tf.shape(fd), 0, 1 / np.sqrt(2)),
                                  tf.random.normal(tf.shape(fd), 0, 1 / np.sqrt(2)))
        return fd + noise_std * noise_normal


class data_split(Layer):
    def __init__(self,  **kwargs):
        super(data_split, self).__init__(**kwargs)

    def call(self, feature, tx, encoded_dim):
        size = tf.shape(feature)
        feature = tf.reshape(feature, [size[0], encoded_dim // tx, tx])
        return feature[:, :, 0], feature[:, :, 1], feature[:, :, 2], feature[:, :, 3]


class data_combin(Layer):
    def __init__(self, args, **kwargs):
        self.tx = args.tx
        super(data_combin, self).__init__(**kwargs)

    def call(self, decoder_out_1, decoder_out_2, decoder_out_3, decoder_out_4):
        rv = tf.concat([decoder_out_1, decoder_out_2, decoder_out_3, decoder_out_4], axis=2)
        shape = tf.shape(rv)
        return tf.reshape(rv, [shape[0], shape[1]*shape[2]])

