from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import argparse
import os
from util_metric import *
from util_norm import *
from util_module import *


def DAnorm_to_HS(x_norm, x_norm_para, carrier_num=256):
    x_trun_c = get_csi_denorm(x_norm, x_norm_para)
    shape = np.shape(x_trun_c)
    x_c = np.concatenate([x_trun_c, np.zeros(shape=[shape[0], 256-shape[1], shape[2]])], axis=1)
    x_DS = np.fft.ifft(x_c, axis=2)
    x_FS = np.fft.fft(x_DS, axis=1)
    return x_FS


def train(args):
    # encoder model
    en_input = Input(shape=(32, 32, 2), name="encoder_input")
    en_output = CsiNetPlusEncoder(en_input, args.encoded_dim)
    en_model = Model(inputs=en_input, outputs=en_output, name='encoder')
    en_model.load_weights('model/CsiNet_plus_ed{}_en_b.h5'.format(args.encoded_dim))
    for layer in en_model.layers:
        layer.trainable = False

    # decoder model
    de_input = Input(shape=(args.encoded_dim,), name='decoder_input')
    de_output = CsiNetPlusDecoder(de_input, residual_num=5)
    de_model = Model(inputs=de_input, outputs=de_output, name="decoder")
    de_model.load_weights('model/CsiNet_plus_ed{}_de_b.h5'.format(args.encoded_dim))

    #offset model
    offset_input = Input(shape=(args.encoded_dim,), name='offset_input')
    offset_output = Offset(offset_input, args.encoded_dim)
    offset_model = Model(inputs=offset_input, outputs=offset_output, name="offset")
    offset_model.load_weights('model/CsiNet_plus_Q_stage1_offset_ed{}_B{}_b.h5'.format(args.encoded_dim, args.B))
    # autoencoder model
    autoencoder_input = Input(shape=(32, 32, 2), name="original_img")
    encoder_out = en_model(autoencoder_input)
    x = MuCompandLayer()(encoder_out)
    x = QuantizationLayer(B=args.B)(x)
    x = DequantizationLayer(B=args.B)(x)
    x = DeMuCompandLayer()(x)
    offset_out = offset_model(x)
    decoder_out = de_model(offset_out)
    autoencoder = Model(inputs=autoencoder_input, outputs=decoder_out, name='autoencoder')
    autoencoder.compile(Adam(args.learning_rate), 'mse')
    autoencoder.summary()

    data_path = 'data_npz/Indoor_tx32_c256_csi_DA_trun.npz'
    
    data = np.load(data_path)
    x_train = data['x_train']
    x_val = data['x_val']
    x_test = data['x_test']
    x_test_norm_para = data['x_test_norm_para']

    filename = os.path.basename(__file__).split('.')[0]
    best_model_path = 'model/{}_ed{}_B{}_b.h5'.format(filename, args.encoded_dim, args.B)
    save_best = ModelCheckpoint(best_model_path, verbose=1, save_best_only=True, save_weights_only=True)
    autoencoder.fit(x_train, x_train,
                    epochs=args.epochs,
                    batch_size=200,
                    validation_data=(x_val, x_val),
                    callbacks=[save_best], verbose=args.verbose)

    # Performance evaluation
    autoencoder.load_weights(best_model_path)
    offset_model.save_weights('model/{}_offset_ed{}_B{}_b.h5'.format(filename, args.encoded_dim, args.B))
    de_model.save_weights('model/{}_de_ed{}_B{}_b.h5'.format(filename, args.encoded_dim, args.B))
    x_pre = autoencoder.predict(x_test)
    x_FS_pre_c = DAnorm_to_HS(x_pre, x_test_norm_para)
    FS_data = np.load('data_npz/Indoor_tx32_c256_csi.npz')
    x_FS_norm = FS_data['x_test']
    x_FS_norm_para = FS_data['x_test_norm_para']
    x_FS_origin_c = get_csi_denorm(x_FS_norm, x_FS_norm_para)
    nmse = cal_nmse(x_FS_origin_c, x_FS_pre_c)
    rho = cal_cosine_similarity(x_FS_origin_c, x_FS_pre_c)
    print("nmse:{}, rho:{}".format(nmse, rho))

def main(args):
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("command", help='train/eval')
    ''' System parameter'''
    parser.add_argument("-ed", "--encoded_dim",
                        help="compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32",
                        default=32, type=int)
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("-B", "--B", default=3, type=int)
    parser.add_argument("-e", "--epochs", default=200, type=int)
    parser.add_argument("-v", "--verbose", default=2, type=int)
    global args
    args = parser.parse_args()
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    main(args)
