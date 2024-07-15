from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import argparse
import os
from util_metric import *
from util_norm import *
from util_module import *
import scipy.io as sio


def DAnorm_to_HS(x_norm, x_norm_para, carrier_num=256):
    x_trun_c = get_csi_denorm(x_norm, x_norm_para)
    shape = np.shape(x_trun_c)
    x_c = np.concatenate([x_trun_c, np.zeros(shape=[shape[0], 256 - shape[1], shape[2]])], axis=1)
    x_DS = np.fft.ifft(x_c, axis=2)
    x_FS = np.fft.fft(x_DS, axis=1)
    return x_FS


def predict(args):
    # encoder model
    en_input = Input(shape=(32, 32, 2), name="encoder_input")
    en_output = CsiNetPlusEncoder(en_input, args.encoded_dim)
    en_model = Model(inputs=en_input, outputs=en_output, name='encoder')
    en_model.load_weights('model/CsiNet_plus_ed{}_en_b.h5'.format(args.encoded_dim))

    qen_input = Input(shape=(32, 32, 2), name='q_en_input')
    x = en_model(qen_input)
    x = MuCompandLayer()(x)
    qen_output = QuantizationLayer(B=args.B)(x)
    qen_model = Model(inputs=qen_input, outputs=qen_output, name="q")

    data_path = 'data_npz/Indoor_tx32_c256_csi_DA_trun.npz'
    
    data = np.load(data_path)
    x_test = data['x_test']
    qen = qen_model.predict(x_test)
    sio.savemat('data_qen/qen_ed{}_B{}.mat'.format(args.encoded_dim, args.B), {'qen': qen})


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
    parser.add_argument("-v", "--verbose", default=2, type=int)
    global args
    args = parser.parse_args()
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    main(args)
