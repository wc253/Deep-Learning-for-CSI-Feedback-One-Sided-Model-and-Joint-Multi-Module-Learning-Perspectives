import numpy as np
from tensorflow.keras.layers import Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation, PReLU, \
    GlobalAveragePooling2D, Concatenate, Multiply, Conv2DTranspose, Layer, Add

import tensorflow as tf


def C2R(x):
    x_r = tf.math.real(x)[..., tf.newaxis]
    x_i = tf.math.imag(x)[..., tf.newaxis]
    x = tf.concat([x_r, x_i], axis=-1)
    return x
    

def add_common_layers(y):
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y


def AF_Module(inputs, snr):
    (_, width, height, ch_num) = inputs.shape
    m = GlobalAveragePooling2D()(inputs)
    m = Concatenate()([m, snr])
    m = Dense(64, activation='relu')(m)
    m = Dense(ch_num, activation='sigmoid')(m)
    out = Multiply()([inputs, m])
    return out


def AFCsiNetPlusEncoder(input_tensor, encoded_dim, snr):
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(input_tensor)
    x = add_common_layers(x)
    x = AF_Module(x, snr)
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(x)
    x = add_common_layers(x)
    x = AF_Module(x, snr)
    x = Reshape((32 * 13 * 2,))(x)
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

    x = Dense(32 * 13 * 2, activation='linear', kernel_initializer='truncated_normal', name='decoded')(input_tensor)
    x = Reshape((32, 13, 2))(x)
    x = AF_Module(x, snr)
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = AF_Module(x, snr)

    for i in range(residual_num):
        x = residual_block_decoded(x, snr)

    x = Activation('relu')(x)
    return x


def CsiNetPlusEncoder(input_tensor, encoded_dim):
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(input_tensor)
    x = add_common_layers(x)
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(x)
    x = add_common_layers(x)
    x = Reshape((32 * 13 * 2,))(x)
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

    x = Dense(32 * 13 * 2, activation='linear', kernel_initializer='truncated_normal', name='decoded')(input_tensor)
    x = Reshape((32, 13, 2))(x)
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    for i in range(residual_num):
        x = residual_block_decoded(x)

    x = Activation('relu')(x)
    return x


class UE2_joint_precoding(Layer):
    def __init__(self):
        self.UE_num = 2
        self.subband_num = 13
        self.tx = 32

        self.FC_11 = Dense(units=1024, activation='relu', kernel_initializer='truncated_normal')
        self.FC_12 = Dense(units=512, activation='relu', kernel_initializer='truncated_normal')
        self.FC_13 = Dense(units=512, activation='relu', kernel_initializer='truncated_normal')
        self.FC_14 = Dense(units=32 * 2, activation='linear', kernel_initializer='truncated_normal')
        self.BN_11 = BatchNormalization()
        self.BN_12 = BatchNormalization()
        self.BN_13 = BatchNormalization()
        self.reshape1 = Reshape((32, 2))

        self.FC_21 = Dense(units=1024, activation='relu', kernel_initializer='truncated_normal')
        self.FC_22 = Dense(units=512, activation='relu', kernel_initializer='truncated_normal')
        self.FC_23 = Dense(units=512, activation='relu', kernel_initializer='truncated_normal')
        self.FC_24 = Dense(units=32 * 2, activation='sigmoid', kernel_initializer='truncated_normal')
        self.BN_21 = BatchNormalization()
        self.BN_22 = BatchNormalization()
        self.BN_23 = BatchNormalization()
        self.reshape2 = Reshape((32, 2))
        super(UE2_joint_precoding, self).__init__()

    def call(self, V1, V2):
        V1_norm = tf.sqrt(tf.reduce_sum(tf.abs(V1) ** 2, axis=[1, 3], keepdims=True))  # 10000*1*13*1
        V1 = tf.divide(V1, V1_norm)
        V2_norm = tf.sqrt(tf.reduce_sum(tf.abs(V2) ** 2, axis=[1, 3], keepdims=True))  # 10000*1*13*1
        V2 = tf.divide(V2, V2_norm)

        input_tensor = tf.concat([V1, V2], axis=1)  # 10000*64*13*2
        for k in range(self.subband_num):
            input_v = input_tensor[:, :, k, :]  # 10000*64*2
            v_r = input_v[..., 0]  # 10000*64*1
            v_i = input_v[..., 1]  # 10000*64*1
            V_r = self.FC_layer_r(v_r)  # 10000*32*2
            V_i = self.FC_layer_i(v_i)  # 10000*32*2
            norm_V = tf.sqrt(tf.reduce_sum(tf.abs(V_r) ** 2 + tf.abs(V_i) ** 2, axis=1, keepdims=True))  # 10000*1*2
            V_r = 1 * tf.divide(V_r, norm_V)
            V_i = 1 * tf.divide(V_i, norm_V)
            power_V = tf.reduce_sum(tf.abs(V_r) ** 2 + tf.abs(V_i) ** 2, axis=1, keepdims=True)  # for check!
            V = tf.expand_dims(tf.complex(V_r, V_i), axis=1)  # 10000*1*32*2
            if k == 0:
                V_matrix = V
            else:
                V_matrix = tf.concat([V_matrix, V], axis=1)  # 10000*13*32*2

        V_1 = V_matrix[:, :, :, 0]
        V_1 = C2R(V_1)
        V_1 = tf.transpose(V_1, perm=[0, 2, 1, 3])  # 10000*32*13*2
        V_2 = V_matrix[:, :, :, 1]
        V_2 = C2R(V_2)  # 10000*13*32*2
        V_2 = tf.transpose(V_2, perm=[0, 2, 1, 3])  # 10000*32*13*2
        mix_V = tf.concat([V_1, V_2], axis=1)
        return mix_V

    def FC_layer_r(self, input_tensor):
        x = self.FC_11(input_tensor)
        x = self.BN_11(x)
        x = self.FC_12(x)
        x = self.BN_12(x)
        x = self.FC_13(x)
        x = self.BN_13(x)
        x = self.FC_14(x)
        out = self.reshape1(x)
        return out

    def FC_layer_i(self, input_tensor):
        x = self.FC_21(input_tensor)
        x = self.BN_21(x)
        x = self.FC_22(x)
        x = self.BN_22(x)
        x = self.FC_23(x)
        x = self.BN_23(x)
        x = self.FC_24(x)
        out = self.reshape2(x)
        return out


class PF_Moudle(Layer):
    def __init__(self, args):
        self.Power_sum = args.P_sum
        self.Num_subbband = args.subband

        self.PF_FC_1 = Dense(64, activation='relu', kernel_initializer='truncated_normal')
        self.PF_BN_1 = BatchNormalization()
        self.PF_FC_2 = Dense(64, activation='relu', kernel_initializer='truncated_normal')
        self.PF_BN_2 = BatchNormalization()
        self.PF_FC_3 = Dense(2, activation='linear', kernel_initializer='truncated_normal')
        self.PF_softmax = Activation('softmax')

        super(PF_Moudle, self).__init__()

    def call(self, input_concat):
        Power_sum = tf.convert_to_tensor(self.Power_sum, dtype=tf.float32)

        for k in range(self.Num_subbband):
            P_ferature_R = input_concat[:,k,:]  # 1000*3
            P_power = self.PF_model_subband_level(P_ferature_R)
            P = P_power * Power_sum
            if k == 0:
                P_matrix = tf.expand_dims(P, axis=1)
            else:
                P = tf.expand_dims(P, axis=1)
                P_matrix = tf.concat([P_matrix, P], axis=1)

        return P_matrix
    def PF_model_subband_level(self, input_tensor):
        x = self.PF_FC_1(input_tensor)
        x = self.PF_BN_1(x)
        x = self.PF_FC_2(x)
        x = self.PF_BN_2(x)
        x = self.PF_FC_3(x)
        x = self.PF_softmax(x)
        return x


class UE2_joint_precoding2(Layer):
    def __init__(self):
        self.UE_num = 2
        self.subband_num = 13
        self.tx = 32
        # FC_joint_precoder
        self.FC_1 = Dense(units=1024, activation='relu', kernel_initializer='truncated_normal')
        self.FC_2 = Dense(units=512, activation='relu', kernel_initializer='truncated_normal')
        self.FC_3 = Dense(units=512, activation='relu', kernel_initializer='truncated_normal')
        self.FC_4r = Dense(units=32 * 2, activation='linear', kernel_initializer='truncated_normal')
        self.FC_4i = Dense(units=32 * 2, activation='linear', kernel_initializer='truncated_normal')
        self.BN_1 = BatchNormalization()
        self.BN_2 = BatchNormalization()
        self.BN_3 = BatchNormalization()
        self.reshape_r = Reshape((32, 2))
        self.reshape_i = Reshape((32, 2))

        super(UE2_joint_precoding2, self).__init__()

    def call(self, V1, V2):
        V1_norm = tf.sqrt(tf.reduce_sum(tf.abs(V1)**2, axis=[1, 3], keepdims=True))  # 10000*1*13*1
        V1 = tf.divide(V1, V1_norm)
        V2_norm = tf.sqrt(tf.reduce_sum(tf.abs(V2)**2, axis=[1, 3], keepdims=True))  # 10000*1*13*1
        V2 = tf.divide(V2, V2_norm)

        V1_r = tf.concat([V1[..., 0], V1[..., 1]], axis=1)  # 100000*64*13
        V2_r = tf.concat([V2[..., 0], V2[..., 1]], axis=1)  # 100000*64*13

        input_tensor = tf.concat([V1_r, V2_r], axis=1)  # 10000*128*13
        for k in range(self.subband_num):
            input_v = input_tensor[:, :, k]  # 10000*128
            V_r, V_i = self.FC_layer(input_v)  # 10000*32*2
            V_c = tf.complex(V_r, V_i)
            norm_V = tf.sqrt(tf.reduce_sum(tf.abs(V_c) ** 2, axis=1, keepdims=True))  # 10000*1*2
            norm_V = tf.cast(norm_V, dtype=V_c.dtype)
            V_c_norm = tf.divide(V_c, norm_V)  # 10000*32*2
            power_V = tf.reduce_sum(tf.abs(V_c_norm) ** 2, axis=1, keepdims=True)  # for check!
            V = tf.expand_dims(V_c_norm, axis=1)  # 10000*1*32*2
            if k == 0:
                V_matrix = V
            else:
                V_matrix = tf.concat([V_matrix, V], axis=1)  # 10000*13*32*2

        V_1 = V_matrix[:, :, :, 0]
        V_1 = C2R(V_1)
        V_1 = tf.transpose(V_1, perm=[0, 2, 1, 3])  # 10000*32*13*2
        V_2 = V_matrix[:, :, :, 1]
        V_2 = C2R(V_2)  # 10000*13*32*2
        V_2 = tf.transpose(V_2, perm=[0, 2, 1, 3])  # 10000*32*13*2
        mix_V = tf.concat([V_1, V_2], axis=1)
        return mix_V

    def FC_layer(self, input_tensor):
        x = self.FC_1(input_tensor)
        x = self.BN_1(x)
        x = self.FC_2(x)
        x = self.BN_2(x)
        x = self.FC_3(x)
        x = self.BN_3(x)
        x_r = self.FC_4r(x)
        x_i = self.FC_4i(x)
        out_r = self.reshape_r(x_r)
        out_i = self.reshape_i(x_i)
        return out_r, out_i
        
        
def up_sample_Decoder(input_tensor):

    x = Dense(32 * 13 * 2, activation='linear', kernel_initializer='truncated_normal', name='decoded')(input_tensor)
    x = Reshape((32, 13, 2))(x)
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    return x
    
    
def up_sample_Decoder2(input_tensor):

    x = Dense(32 * 13 * 2, activation='linear', kernel_initializer='truncated_normal', name='decoded')(input_tensor)
    x = Reshape((32, 13, 2))(x)
    x = Activation('sigmoid')(x)

    return x
    
    
class E2E_decoder(Layer):
    def __init__(self):
        self.UE_num = 2
        self.subband_num = 13
        self.tx = 32
        # FC_joint_precoder
        self.FC_1 = Dense(units=1024, activation='relu', kernel_initializer='truncated_normal')
        self.FC_2 = Dense(units=512, activation='relu', kernel_initializer='truncated_normal')
        self.FC_3 = Dense(units=512, activation='relu', kernel_initializer='truncated_normal')
        self.FC_4_1 = Dense(units=32 * 13, activation='relu', kernel_initializer='truncated_normal')
        self.FC_4_2 = Dense(units=32 * 13, activation='relu', kernel_initializer='truncated_normal')
        self.FC_UE1_r = Dense(units=32 * 13, activation='linear', kernel_initializer='truncated_normal')
        self.FC_UE1_i = Dense(units=32 * 13, activation='linear', kernel_initializer='truncated_normal')
        self.FC_UE2_r = Dense(units=32 * 13, activation='linear', kernel_initializer='truncated_normal')
        self.FC_UE2_i = Dense(units=32 * 13, activation='linear', kernel_initializer='truncated_normal')

        self.BN_1 = BatchNormalization()
        self.BN_2 = BatchNormalization()
        self.BN_3 = BatchNormalization()
        self.BN_4_1 = BatchNormalization()
        self.BN_4_2 = BatchNormalization()

        self.reshape_1_r = Reshape((32, 13))
        self.reshape_1_i = Reshape((32, 13))
        self.reshape_2_r = Reshape((32, 13))
        self.reshape_2_i = Reshape((32, 13))

        super(E2E_decoder, self).__init__()

    def call(self, rv1, rv2):
        x = tf.concat([rv1, rv2], axis=-1)
        V_1_r, V_1_i, V_2_r, V_2_i = self.FC_layer(x)
        V1 = tf.complex(V_1_r, V_1_i)
        V2 = tf.complex(V_2_r, V_2_i)
        V1_norm = tf.sqrt(tf.reduce_sum(tf.abs(V1)**2, axis=1, keepdims=True))  # 10000*1*13
        V1_norm = tf.cast(V1_norm, dtype=V1.dtype)
        V1 = tf.divide(V1, V1_norm)
        V2_norm = tf.sqrt(tf.reduce_sum(tf.abs(V2)**2, axis=1, keepdims=True))  # 10000*1*13
        V2_norm = tf.cast(V2_norm, dtype=V1.dtype)
        V2 = tf.divide(V2, V2_norm)

        V_1 = C2R(V1)  # 10000*32*13*2
        V_2 = C2R(V2)  # 10000*32*13*2
        mix_V = tf.concat([V_1, V_2], axis=1)
        return mix_V

    def FC_layer(self, input_tensor):
        x = self.FC_1(input_tensor)
        x = self.BN_1(x)
        x = self.FC_2(x)
        x = self.BN_2(x)
        x = self.FC_3(x)
        x = self.BN_3(x)
        x_1 = self.FC_4_1(x)
        x_1 = self.BN_4_1(x)
        x_2 = self.FC_4_2(x)
        x_2 = self.BN_4_2(x)
        x_1_r = self.FC_UE1_r(x_1)
        x_1_i = self.FC_UE1_i(x_2)
        V_1_r = self.reshape_1_r(x_1_r)
        V_1_i = self.reshape_1_i(x_1_i)
        x_2_r = self.FC_UE2_r(x_2)
        x_2_i = self.FC_UE2_i(x_2)
        V_2_r = self.reshape_2_r(x_2_r)
        V_2_i = self.reshape_2_i(x_2_i)
        return V_1_r, V_1_i, V_2_r, V_2_i
