import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import argparse, json
from util_module import *
from util_channel import UplinkChannelMRC
from util_norm import *
from util_metric import *

def main(args):
    # encoder model
    en_input = Input(shape=(256, 32, 2), name="encoder_input")
    en_snrdb = Input(shape=(1,))
    en_t = AFTransEncoder(en_input, en_snrdb)
    en_output = AFCsiNetPlusEncoder(en_t, args.encoded_dim, en_snrdb)
    en_model = Model(inputs=[en_input, en_snrdb], outputs=en_output, name='encoder')
    # decoder model
    de_input = Input(shape=(args.encoded_dim,), name='decoder_input')
    de_snrdb = Input(shape=(1,))
    de_output = AFCsiNetPlusDecoder(de_input, de_snrdb, residual_num=5)
    de_output = AFTransDecoder(de_output, de_snrdb)
    de_model = Model(inputs=[de_input, de_snrdb], outputs=de_output, name="decoder")

    # autoencoder model
    input_csi = Input(shape=(256, 32, 2), name="original_img")
    input_h = Input(shape=(args.encoded_dim//2, 32, 2))
    input_snr = Input(shape=(1,))
    encoder_out = en_model([input_csi, input_snr])
    rv = UplinkChannelMRC()(encoder_out, input_h, input_snr)
    decoder_out = de_model([rv, input_snr])
    model = Model(inputs=[input_csi, input_h, input_snr], outputs=decoder_out, name='autoencoder')
    model.compile(Adam(args.learning_rate), loss='mse')
    model.summary()


    data = np.load('data_npz/Indoor_tx32_c256_csi.npz')
    x_train = data['x_train']
    x_val = data['x_val']
    x_test = data['x_test']
    x_test_norm_para = data['x_test_norm_para']
    

    data = np.load('data_npz/Indoor_tx32_c256_H.npz')
    x_train_h = data['x_train']
    x_val_h = data['x_val']
    x_test_h = data['x_test']
    x_train_h = x_train_h[:, 0:args.encoded_dim // 2, :, :]
    x_val_h = x_val_h[:, 0:args.encoded_dim // 2, :, :]
    x_test_h = x_test_h[:, 0:args.encoded_dim // 2, :, :]

    x_train_snr = np.random.uniform(args.snr_low, args.snr_high, size=(np.shape(x_train)[0],))
    x_val_snr = np.random.uniform(args.snr_low, args.snr_high, size=(np.shape(x_val)[0],))

    filename = os.path.basename(__file__).split('.')[0]
    best_path = 'model/{}_ed{}_snr{}to{}_b.h5'.format(filename, args.encoded_dim, args.snr_low, args.snr_high)
    best_model = ModelCheckpoint(filepath=best_path, save_weights_only=True, save_best_only=True, verbose=1)
    lr_reduce = ReduceLROnPlateau(factor=0.5, patience=20, verbose=1, min_delta=0, min_lr=1e-4)
    model.fit(x=(x_train, x_train_h, x_train_snr), y=x_train, batch_size=200, epochs=args.epochs,
              validation_data=((x_val, x_val_h, x_val_snr), x_val),
              callbacks=[best_model, lr_reduce], verbose=args.verbose)

    # performance evaluation
    model.load_weights(best_path)
    x_test_c = get_csi_denorm(x_test, x_test_norm_para)
    snr_list = []
    nmse_list = []
    rho_list = []
    for snr in range(args.snr_low, args.snr_high + 1, 1):
        x_test_snr = snr * np.ones([np.shape(x_test)[0], ])
        x_pre = model.predict([x_test, x_test_h, x_test_snr])
        x_pre_c = get_csi_denorm(x_pre, x_test_norm_para)
        nmse = cal_nmse(x_test_c, x_pre_c)
        rho = cal_cosine_similarity(x_test_c, x_pre_c)
        snr_list.append(snr)
        nmse_list.append(nmse)
        rho_list.append(rho)
        print('snr:{}, nmse:{}, rho:{}'.format(snr, nmse, rho))
    with open('data_eval/{}_ed{}_snr{}to{}.json'.format(filename, args.encoded_dim, args.snr_low, args.snr_high),
              mode='w') as f:
        json.dump({'snr': snr_list, 'nmse': nmse_list, 'rho': rho_list}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ''' System parameter'''
    parser.add_argument("-ed", "--encoded_dim", default=32, type=int)
    parser.add_argument("-sl", "--snr_low", default=-10, type=int)
    parser.add_argument("-sh", "--snr_high", default=10, type=int)
    parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float)
    parser.add_argument("-e", "--epochs", default=500, type=int)
    parser.add_argument("-v", "--verbose", default=1, type=int)
    global args
    args = parser.parse_args()
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    main(args)
