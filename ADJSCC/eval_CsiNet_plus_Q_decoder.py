from scipy.io import loadmat
import numpy as np
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import argparse
import os
from util_metric import *
from util_norm import *
from util_module import *
import json


def DAnorm_to_HS(x_norm, x_norm_para, carrier_num=256):
    x_trun_c = get_csi_denorm(x_norm, x_norm_para)
    shape = np.shape(x_trun_c)
    x_c = np.concatenate([x_trun_c, np.zeros(shape=[shape[0], 256 - shape[1], shape[2]])], axis=1)
    x_DS = np.fft.ifft(x_c, axis=2)
    x_FS = np.fft.fft(x_DS, axis=1)
    return x_FS


def predict(args):
     # decoder model
    de_input = Input(shape=(args.encoded_dim,), name='decoder_input')
    de_output = CsiNetPlusDecoder(de_input, residual_num=5)
    de_model = Model(inputs=de_input, outputs=de_output, name="decoder")
    de_model.load_weights('model/CsiNet_plus_Q_stage2_de_ed{}_B{}_b.h5'.format(args.encoded_dim, args.B))

    #offset model
    offset_input = Input(shape=(args.encoded_dim,), name='offset_input')
    offset_output = Offset(offset_input, args.encoded_dim)
    offset_model = Model(inputs=offset_input, outputs=offset_output, name="offset")
    offset_model.load_weights('model/CsiNet_plus_Q_stage2_offset_ed{}_B{}_b.h5'.format(args.encoded_dim, args.B))

    qde_input = Input(shape=(args.encoded_dim,), name='qde_input')
    x = DequantizationLayer(B=args.B)(qde_input)
    x = DeMuCompandLayer()(x)
    offset_out = offset_model(x)
    qde_output = de_model(offset_out)
    qde_model = Model(inputs=qde_input, outputs=qde_output)
    
    data_path = 'data_npz/Indoor_tx32_c256_csi_DA_trun.npz'
    
    data = np.load(data_path)
    x_test_norm_para = data['x_test_norm_para']
    
    data_path = 'data_npz/Indoor_tx32_c256_csi.npz'
    
    FS_data = np.load(data_path)
    
    x_FS_norm = FS_data['x_test']
    x_FS_norm_para = FS_data['x_test_norm_para']
    x_FS_origin_c = get_csi_denorm(x_FS_norm, x_FS_norm_para)

    snr_list = []
    nmse_list = []
    rho_list = []
    for snr in range(-1,1):
        filename = 'data_dec/dec_bw{}_ed{}_B{}_mod{}_snr{}.mat'.format(args.bw, args.encoded_dim, args.B, args.mod, snr)
        data_mat = loadmat(filename)
        dec = data_mat['dec']
        x_pre = qde_model.predict(dec, verbose=args.verbose)
        x_FS_pre_c = DAnorm_to_HS(x_pre, x_test_norm_para)

        nmse = cal_nmse(x_FS_origin_c, x_FS_pre_c)
        rho = cal_cosine_similarity(x_FS_origin_c, x_FS_pre_c)
        snr_list.append(snr)
        nmse_list.append(nmse)
        rho_list.append(rho)
        print("snr:{}, nmse:{}, rho:{}".format(snr, nmse, rho))
    with open('data_eval/bw{}_ed{}_B{}_mod{}.json'.format(args.bw, args.encoded_dim, args.B, args.mod),
              mode='w') as f:
        json.dump({'snr': snr_list, 'nmse': nmse_list, 'rho': rho_list}, f)


def main(args):
    predict(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("command", help='train/eval')
    ''' System parameter'''
    parser.add_argument("-ed", "--encoded_dim",
                        help="compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32",
                        default=32, type=int)
    parser.add_argument("-B", "--B", default=3, type=int)
    parser.add_argument("-bw", "--bw", default=16, type=int)
    parser.add_argument("-m", "--mod", default=8, type=int)
    parser.add_argument("-v", "--verbose", default=1, type=int)
    global args
    args = parser.parse_args()
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    main(args)
