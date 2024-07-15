import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import argparse
import os
from util_metric import *

def add_common_layers(y):
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    return y


def Encoder(input_tensor, encoded_dim):
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(input_tensor)
    x = add_common_layers(x)
    x = Conv2D(2, (7, 7), padding='same', kernel_initializer='truncated_normal')(x)
    x = add_common_layers(x)
    x = Reshape((32 * 32 * 2,))(x)
    encoded = Dense(encoded_dim, activation='tanh', kernel_initializer='truncated_normal', name='encoded')(x)
    return encoded


def Decoder(input_tensor, residual_num=5):
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


def train(args):
    en_input = Input(shape=(32, 32, 2), name="encoder_input")
    en_output = Encoder(en_input, args.encoded_dim)
    en_model = Model(inputs=en_input, outputs=en_output, name='encoder')

    # decoder model
    de_input = Input(shape=(args.encoded_dim,), name='decoder_input')
    de_output = Decoder(de_input, residual_num=5)
    de_model = Model(inputs=de_input, outputs=de_output, name="decoder")

    # autoencoder model
    autoencoder_input = Input(shape=(32, 32, 2), name="original_img")
    encoder_out = en_model(autoencoder_input)
    decoder_out = de_model(encoder_out)
    autoencoder = Model(inputs=autoencoder_input, outputs=decoder_out, name='autoencoder')
    autoencoder.compile(Adam(args.learning_rate), loss='mse')
    autoencoder.summary()

    data_path = 'data_npz/Indoor_tx32_c256_csi_DA_trun.npz'
    data = np.load(data_path)
    x_train = data['x_train']
    x_val = data['x_val']
    x_test = data['x_test']

    filename = os.path.basename(__file__).split('.')[0]
    best_model_path = 'model/{}_ed{}_b.h5'.format(filename, args.encoded_dim)
    save_best = ModelCheckpoint(best_model_path, verbose=1, save_best_only=True, save_weights_only=True)
    lr_reduce = ReduceLROnPlateau(factor=0.5, patience=20, verbose=1, min_delta=0, min_lr=1e-4)
    autoencoder.fit(x_train, x_train,
                    epochs=args.epochs,
                    batch_size=200,
                    validation_data=(x_val, x_val),
                    callbacks=[save_best, lr_reduce], verbose=args.verbose)

    autoencoder.load_weights(best_model_path)
    en_model.save_weights('model/{}_ed{}_en_b.h5'.format(filename, args.encoded_dim))
    de_model.save_weights('model/{}_ed{}_de_b.h5'.format(filename, args.encoded_dim))
    # metric on delay spread domain

    x_hat = autoencoder.predict(x_test)
    x_test_c = (x_test[:, :, :, 0] - 0.5) + 1j * (x_test[:, :, :, 1] - 0.5)
    x_hat_c = (x_hat[:, :, :, 0] - 0.5) + 1j * (x_hat[:, :, :, 1] - 0.5)
    nmse = cal_nmse(x_test_c, x_hat_c)
    rho = cal_cosine_similarity(x_test_c, x_hat_c)
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
    parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float)
    parser.add_argument("-e", "--epochs", default=1000, type=int)
    parser.add_argument("-v", "--verbose", default=1, type=int)
    global args
    args = parser.parse_args()
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    main(args)
