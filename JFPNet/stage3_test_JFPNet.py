import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# from keras.layers import Input
# from keras import Model
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import argparse, json
from util_module import *
from util_channel import *
from util_norm import *
from util_metric import *
from util_loss import *
from util_plot import *


# 2UE

def c2r(x):
    x_r = np.real(x)
    x_r = x_r[..., np.newaxis]
    x_i = np.real(x)
    x_i = x_i[..., np.newaxis]
    x_c = np.concatenate([x_r, x_i], axis=-1)
    return x_c


def main(args):
    # ------------------------------------------------------ #
    #                    load datasets                       #
    # ------------------------------------------------------ #
    # load downlink H_RB_level
    data_path = '/gpu05/guoyiran/DJSCC_uma/data_npz/MU_MIMO_UMa_UE0_H_ul_RB.npz'
    # data_path = 'H:/ZTE_ADJSCC/data_npz/MU_MIMO_UMa_UE0_H_ul_RB_less.npz'
    data = np.load(data_path)

    RB_test_1 = data['x_test']

    RB_test_1 = np.transpose(RB_test_1, (0, 1, 3, 2, 4))

    data_path = '/gpu05/guoyiran/DJSCC_uma/data_npz/MU_MIMO_UMa_UE1_H_ul_RB.npz'
    # data_path = 'H:/ZTE_ADJSCC/data_npz/MU_MIMO_UMa_UE1_H_ul_RB_less.npz'
    data = np.load(data_path)

    RB_test_2 = data['x_test']

    RB_test_2 = np.transpose(RB_test_2, (0, 1, 3, 2, 4))

    # load uplink channel matrix
    data_path = '/gpu05/guoyiran/DJSCC_uma/data_npz/MU_MIMO_UMa_UE0_H_ul.npz'
    # data_path = 'H:/ZTE_ADJSCC/data_npz/MU_MIMO_UMa_UE0_H_ul_less.npz'
    data = np.load(data_path)

    x_test_h = data['x_test']

    x_test_h1 = x_test_h[:, 0:args.encoded_dim // 8, :, :, :]

    x_test_h1 = np.transpose(x_test_h1, (0, 1, 3, 2, 4))
    data_path = '/gpu05/guoyiran/DJSCC_uma/data_npz/MU_MIMO_UMa_UE1_H_ul.npz'
    # data_path = 'H:/ZTE_ADJSCC/data_npz/MU_MIMO_UMa_UE1_H_ul_less.npz'
    data = np.load(data_path)

    x_test_h = data['x_test']

    x_test_h2 = x_test_h[:, 0:args.encoded_dim // 8, :, :, :]

    x_test_h2 = np.transpose(x_test_h2, (0, 1, 3, 2, 4))

    # load downlink CSI matrix
    data_path = '/gpu05/guoyiran/DJSCC_uma/data_npz/MU_MIMO_UMa_UE6_H_dl_vector.npz'
    # data_path = 'H:/ZTE_ADJSCC/data_npz/MU_MIMO_UMa_UE6_H_dl_vector_less.npz'
    data = np.load(data_path)
    x_test = data['x_test']
    x_test_norm_para = data['x_test_norm_para']

    eig_test = data['test_eig']

    x_test_1 = x_test[:, 0, ...]
    x_test_norm_para_1 = x_test_norm_para[:, 0, ...]

    eig_test_1 = eig_test[:, 0, :]

    x_test_2 = x_test[:, 1, ...]
    x_test_norm_para_2 = x_test_norm_para[:, 1, ...]

    eig_test_2 = eig_test[:, 1, :]

    # 生成用于PF模块的数据

    noise_test = args.noise_power * np.ones(shape=eig_test_1.shape)

    def PF_input_norm(eig1, eig2, noise):
        eig1 = np.log10(eig1[..., np.newaxis])
        eig2 = np.log10(eig2[..., np.newaxis])
        noise = np.log10(noise[..., np.newaxis])
        x_norm_c = np.concatenate([eig1, eig2, noise], axis=-1)  # 100000*13*3
        norm_para = np.sqrt(np.sum(np.square(x_norm_c), axis=-1, keepdims=True))  # 10000*13*1
        x_norm_c = 1 + x_norm_c / norm_para
        return x_norm_c


    PF_input_test = PF_input_norm(eig_test_1, eig_test_2, noise_test)


    del x_test
    del x_test_norm_para
    del eig_test

    # 生成用于测试的数据:使用完美的V、平均功率分配算R_sum
    x_test1_denorm = get_csi_denorm(x_test_1, x_test_norm_para_1)
    x_test2_denorm = get_csi_denorm(x_test_2, x_test_norm_para_2)
    V_base_c = np.concatenate([x_test1_denorm[..., np.newaxis], x_test2_denorm[..., np.newaxis]], axis=-1)

    # P_dis = eig_train_2 / (eig_train_1 + eig_train_2)
    P_dis = 0.5 * np.ones(shape=eig_test_1.shape)
    P_test1 = (args.P_sum) * P_dis
    P_test2 = (args.P_sum) * (1.0 - P_dis)
    P_test = np.concatenate([P_test1[..., np.newaxis], P_test2[..., np.newaxis]], axis=-1)



    # ------------------------------------------------------ #
    #                     design model                       #
    # ------------------------------------------------------ #

    # encoder model 1
    en_input1 = Input(shape=(32, 13, 2), name="encoder_input1")
    en_output1 = CsiNetPlusEncoder(en_input1, args.encoded_dim)
    en_model1 = Model(inputs=en_input1, outputs=en_output1, name='encoder_1')
    # encoder model 2
    en_input2 = Input(shape=(32, 13, 2), name="encoder_input1")
    en_output2 = CsiNetPlusEncoder(en_input2, args.encoded_dim)
    en_model2 = Model(inputs=en_input2, outputs=en_output2, name='encoder_2')

    # decoder model 1
    de_input_1 = Input(shape=(args.encoded_dim,), name='decoder_input_UE1')
    de_output1 = CsiNetPlusDecoder(de_input_1)
    de_model1 = Model(inputs=de_input_1, outputs=de_output1, name="decoder_1")

    # decoder model 2
    de_input_2 = Input(shape=(args.encoded_dim,), name='decoder_input_UE2')
    de_output2 = CsiNetPlusDecoder(de_input_2)
    de_model2 = Model(inputs=de_input_2, outputs=de_output2, name="decoder_2")

    # joint_precoder
    Precode_input_1 = Input(shape=(32, 13, 2), name='est_V1')
    Precode_input_2 = Input(shape=(32, 13, 2), name='est_V2')
    mix_out = UE2_joint_precoding2()(Precode_input_1, Precode_input_2)  # B*64*13*2
    Joint_Precoder_model = Model(inputs=[Precode_input_1, Precode_input_2], outputs=mix_out, name='joint_precoder')

    # PF_model
    PF_input = Input(shape=(13, 3), name='concat_noise_P')
    PF_out = PF_Moudle(args)(PF_input)
    PF_model = Model(inputs=PF_input, outputs=PF_out, name='PF_model')  # B*13*2

    # DJSCC_model1
    djscc_in_csi = Input(shape=(32, 13, 2))
    djscc_h = Input(shape=(args.encoded_dim // 8, 32, 4, 2))
    djscc_in_snr = Input(shape=(1,))

    encoder_out = en_model1(djscc_in_csi)
    tx1, tx2, tx3, tx4 = data_split()(encoder_out, args.tx, args.encoded_dim)
    rv1, rv2, rv3, rv4 = UplinkChannelMMSE(args)(tx1, tx2, tx3, tx4, djscc_h, djscc_in_snr)
    rv = data_combin(args)(rv1, rv2, rv3, rv4)
    recover_V = de_model1(rv)
    DJSCC_model = Model(inputs=[djscc_in_csi, djscc_h, djscc_in_snr], outputs=recover_V, name='DJSCC_network')


    # autoencoder model
    input_csi_1 = Input(shape=(32, 13, 2), name="UE1_original_img")
    input_csi_2 = Input(shape=(32, 13, 2), name="UE2_original_img")
    input_RB_1 = Input(shape=(52, 32, 4, 2), name="UE1_RB_level_CSI")
    input_RB_2 = Input(shape=(52, 32, 4, 2), name="UE2_RB_level_CSI")
    input_concat = Input(shape=(13, 3), name='concat_PF_input')

    PF_out = PF_model(input_concat)
    out_combin = Joint_Precoder_model([input_csi_1, input_csi_2])
    loss = DL_R_sum_MRC(args)([input_RB_1, input_RB_2, PF_out, out_combin])

    model = Model(inputs=[input_csi_1, input_csi_2,
                          input_RB_1, input_RB_2,
                          input_concat],
                  outputs=[loss, out_combin], name='autoencoder')
    model.compile(Adam(args.learning_rate), loss=None)
    model.summary()

    filename = os.path.basename(__file__).split('.')[0]

    # performance evaluation



    print(filename)
    R_sum_list = []
    R_sum_base_list = []
    R_1_list = []
    R_1_base_list = []
    R_2_list = []
    R_2_base_list = []
    UL_SNR_list = []
    for snr_ul in range(-10, 11, 5):
        x_test_snr_dl = snr_ul * np.ones([np.shape(x_test_1)[0], ])
        # predict reconstruct V
        DJSCC_model.load_weights('model/DJSCC_UE1_ed{}_snr-10to10.h5'.format(args.encoded_dim))
        csi_hat_1 = DJSCC_model.predict([x_test_1, x_test_h1, x_test_snr_dl])
        DJSCC_model.load_weights('model/DJSCC_UE2_ed{}_snr-10to10.h5'.format(args.encoded_dim))
        csi_hat_2 = DJSCC_model.predict([x_test_2, x_test_h2, x_test_snr_dl])
        model.load_weights('model/stage2_JMP_ed{}_snr-10to10_b.h5'.format(args.encoded_dim))
        _, V_pre = model.predict([csi_hat_1, csi_hat_2,
                                  RB_test_1, RB_test_2,
                                  PF_input_test])
        V_pre_1 = V_pre[:, :32, :, :]
        V_pre_2 = V_pre[:, 32:, :, :]
        V_pre_1 = V_pre_1[..., 0] + 1j * V_pre_1[..., 1]
        V_pre_2 = V_pre_2[..., 0] + 1j * V_pre_2[..., 1]
        V_pre = np.concatenate([V_pre_1[..., np.newaxis], V_pre_2[..., np.newaxis]], axis=-1)
        # predict power distribution matrix
        PF_test = PF_model.predict(PF_input_test)

        R_1, R_2, R_Sum = cal_downlink_R_sum(V_pre, RB_test_1, RB_test_2, PF_test, args.noise_power)
        R_1_base, R_2_base, R_Sum_base = cal_downlink_R_sum(V_base_c, RB_test_1, RB_test_2, P_test, args.noise_power)
        UL_SNR_list.append(snr_ul)
        R_1_list.append(R_1)
        R_2_list.append(R_2)
        R_sum_list.append(R_Sum)
        R_1_base_list.append(R_1_base)
        R_2_base_list.append(R_2_base)
        R_sum_base_list.append(R_Sum_base)

        print(
            'UL_snr:{}, tx_P_sum:{}dBm, noise_power:{}dBm, R_Sum:{}, R_Sum_base:{}, R_1:{},R_1_base:{},R_2:{},R_2_base:{}'
            .format(snr_ul, num2dBm(args.P_sum), num2dBm(args.noise_power), R_Sum, R_Sum_base, R_1, R_1_base, R_1,
                    R_1_base))

    with open(
            'data_eval/{}_ed{}_UL_snr{}to{}_Ptx{}dBm_N_{}dBm.json'.format(filename, args.encoded_dim, args.snr_low,
                                                                          args.snr_high, num2dBm(args.P_sum),
                                                                          num2dBm(args.noise_power)),
            mode='w') as f:
        json.dump({'UL_snr': UL_SNR_list, 'R_Sum': R_sum_list, 'R_Sum_base': R_sum_base_list,
                   'R_1': R_1_list, 'R_1_base': R_1_base_list, 'R_2': R_2_list, 'R_2_base': R_2_base_list}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tx", "--tx", default=4, type=int)
    parser.add_argument("-rx", "--rx", default=32, type=int)
    parser.add_argument("-RB", "--RB", default=52, type=int)
    parser.add_argument("-P1", "--P1", default=1, type=int)
    parser.add_argument("-P2", "--P2", default=1, type=int)
    parser.add_argument("-P3", "--P3", default=1, type=int)
    parser.add_argument("-P4", "--P4", default=1, type=int)
    parser.add_argument("-UE", "--UE_num", default=2, type=int)
    parser.add_argument("-subband", "--subband", default=13, type=int)
    parser.add_argument("-P", "--P_sum", default=10 ** 1.6, type=float)  # 10 ** 1.6
    parser.add_argument("--noise_power", default=10 ** (-13.4), type=float)  # 10**(-13.4)
    parser.add_argument("-k", "--encoded_dim", default=32, type=int)
    parser.add_argument("-sl", "--snr_low", default=-10, type=int)
    parser.add_argument("-sh", "--snr_high", default=10, type=int)
    parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float)
    parser.add_argument("-e", "--epochs", default=1, type=int)
    parser.add_argument("-v", "--verbose", default=1, type=int)
    global args
    args = parser.parse_args()
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    main(args)
