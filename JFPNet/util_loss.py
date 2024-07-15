import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Flatten
from tensorflow.python.framework.ops import disable_eager_execution


class DL_R_sum_MRC(Layer):
    def __init__(self, args):
        super(DL_R_sum_MRC, self).__init__()
        self.noise_power = args.noise_power

    def cal_R_sum(self, H_dl_RB_1, H_dl_RB_2, P_marix, y_pred):
        new_V_1 = y_pred[:, :32, :, :]  # 10000*32*13*2
        new_V_2 = y_pred[:, 32:, :, :]  # 10000*32*13*2
        H_dl_1 = tf.complex(H_dl_RB_1[..., 0], H_dl_RB_1[..., 1])[..., tf.newaxis]  # 100000*52*32*4*1
        H_dl_2 = tf.complex(H_dl_RB_2[..., 0], H_dl_RB_2[..., 1])[..., tf.newaxis]  # 100000*52*32*4*1
        V_1_c = tf.complex(new_V_1[..., 0], new_V_1[..., 1])[..., tf.newaxis]  # 100000*32*13*1
        V_2_c = tf.complex(new_V_2[..., 0], new_V_2[..., 1])[..., tf.newaxis]  # 100000*32*13*1
        V_multi = tf.concat([V_1_c, V_2_c], axis=-1)  # 100000*32*13*2
        H_dl = tf.concat([H_dl_1, H_dl_2], axis=-1)  # 100000*52*32*4*2

        for k_index in range(2):
            H_RB = tf.squeeze(H_dl[..., k_index])  # 100000*52*32*4
            for i in range(13):  # 计算13个子带的和
                for j in range(4):  # 计算每个子带中4个RB的和
                    P = tf.squeeze(P_marix[:, i, :])  # 100000*2
                    H = tf.transpose(H_RB[:, 4 * i + j, :, :], [0, 2, 1])  # 100000*4*32
                    F = tf.squeeze(V_multi[:, :, i, k_index])[..., tf.newaxis]  # 100000*32*1
                    norm_F = tf.sqrt(tf.reduce_sum(tf.abs(F) ** 2, axis=1, keepdims=True))  # 10000*1
                    norm_F = tf.complex(norm_F, 0 * norm_F)
                    F = tf.divide(F, norm_F)  # 100000*32*1
                    HF = tf.matmul(H, F)  # 100000*4*1
                    HF_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(HF)), axis=1, keepdims=True))
                    HF_norm_c = tf.cast(HF_norm, dtype=HF.dtype)
                    W = tf.divide(HF, HF_norm_c)
                    for kk in range(2):  # 计算k_index用户在每个RB上的R
                        V = tf.squeeze(V_multi[:, :, i, kk])[..., tf.newaxis]  # 100000*32*1
                        norm_V = tf.sqrt(tf.reduce_sum(tf.pow(tf.abs(V), 2), axis=1, keepdims=True))  # 10000*1
                        norm_V = tf.complex(norm_V, 0 * norm_V)
                        V = tf.divide(V, norm_V)
                        H_V = tf.matmul(H, V)  # 100000*4*1
                        W_H = tf.math.conj(tf.transpose(W, [0, 2, 1]))  # 100000*1*4
                        W_H_H_V = tf.matmul(W_H, H_V)  # 100000*1
                        WHVVHW = tf.square(abs(W_H_H_V))
                        norm = tf.squeeze(WHVVHW) * tf.cast(P[:, kk], dtype=WHVVHW.dtype)
                        norm = tf.cast(norm, dtype=tf.float64)
                        # norm = tf.cast(tf.squeeze(WHVVHW), dtype=W.dtype) * tf.cast(P[:, kk], dtype=W.dtype)
                        if kk == k_index:
                            nom = norm
                        if kk == 0:
                            # noise = tf.cast(self.noise_power, dtype=norm.dtype)
                            # eye_matrix = tf.eye(4, 4, batch_shape=[tf.shape(norm)[0]], dtype=norm.dtype)
                            # noise_matrix = tf.multiply(noise[..., tf.newaxis], eye_matrix)
                            # WZZ = tf.matmul(W_H, noise_matrix)
                            # WZZW = tf.cast(tf.squeeze(tf.matmul(WZZ, W)), dtype=norm.dtype)
                            # nom_denom = norm + WZZW
                            noise = tf.cast(self.noise_power, dtype=tf.float64)
                            nom_denom = norm + noise
                        else:
                            nom_denom = nom_denom + norm

                    denom = tf.cast(tf.math.subtract(nom_denom, nom), dtype=tf.float64)
                    rate = tf.math.log(1 + tf.divide(tf.cast(nom, dtype=tf.float64), denom)) / tf.cast(tf.math.log(2.0),
                                                                                                       dtype=tf.float64)

                    if i == 0:
                        if j == 0:
                            R_ = rate
                        elif 0 < j < 3:
                            R_ = R_ + rate
                        else:
                            R_ = R_ + rate
                            R = R_[..., tf.newaxis]
                    else:
                        if j == 0:
                            R_ = rate
                        elif 0 < j < 3:
                            R_ = R_ + rate
                        else:
                            R_ = R_ + rate
                            R_ = R_[..., tf.newaxis]
                            R = tf.concat([R, R_], axis=-1)

            if k_index == 0:
                R_sum = tf.reduce_mean(R, axis=-1) / 4
            else:
                R_2 = tf.reduce_mean(R, axis=-1) / 4
                R_sum = R_sum + R_2

        return -tf.reduce_mean(R_sum)

    def call(self, inputs):
        H_dl_RB_1 = inputs[0]
        H_dl_RB_2 = inputs[1]
        P_marix = inputs[2]
        y_pred = inputs[3]
        loss = self.cal_R_sum(H_dl_RB_1, H_dl_RB_2, P_marix, y_pred)
        # print_loss = tf.print(loss, "-->loss")
        self.add_loss(loss, inputs=inputs)
        return y_pred

