import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import argparse
from utils.util_module import *
from utils.util_metric import *
from utils.util_norm import *
from utils.util_pnp import *
import random
import math
import time
from scipy.io import savemat
import matplotlib.pyplot as plt

# seed
seed = 100
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)


def block_solver(VT, encode_dim):
    V11V11T = np.matmul(np.transpose(VT[:, 0:encode_dim, 0:encode_dim], (0, 2, 1)), VT[:, 0:encode_dim, 0:encode_dim])
    V12V12T = np.matmul(np.transpose(VT[:, encode_dim:2048, 0:encode_dim], (0, 2, 1)),
                        VT[:, encode_dim:2048, 0:encode_dim])
    V11V21T = np.matmul(np.transpose(VT[:, 0:encode_dim, 0:encode_dim], (0, 2, 1)),
                        VT[:, 0:encode_dim, encode_dim:2048])
    V12V22T = np.matmul(np.transpose(VT[:, encode_dim:2048, 0:encode_dim], (0, 2, 1)),
                        VT[:, encode_dim:2048, encode_dim:2048])
    V21V21T = np.matmul(np.transpose(VT[:, 0:encode_dim, encode_dim:2048], (0, 2, 1)),
                        VT[:, 0:encode_dim, encode_dim:2048])
    V22V22T = np.matmul(np.transpose(VT[:, encode_dim:2048, encode_dim:2048], (0, 2, 1)),
                        VT[:, encode_dim:2048, encode_dim:2048])
    return V11V11T, V12V12T, V11V21T, V12V22T, V21V21T, V22V22T


def svd_solver(V11V11T, V12V12T, V11V21T, V12V22T, V21V21T, V22V22T, encode_dim, rho):
    P_inv = np.zeros((100, 2048, 2048))
    P_inv[:, 0:encode_dim, 0:encode_dim] = np.dot(1 / (2 + rho), V11V11T) + np.dot(1 / rho, V12V12T)
    P_inv[:, 0:encode_dim, encode_dim:2048] = np.dot(1 / (2 + rho), V11V21T) + np.dot(1 / rho, V12V22T)
    P_inv[:, encode_dim:2048, 0:encode_dim] = np.transpose(P_inv[:, 0:encode_dim, encode_dim:2048], (0, 2, 1))
    P_inv[:, encode_dim:2048, encode_dim:2048] = np.dot(1 / (2 + rho), V21V21T) + np.dot(1 / rho, V22V22T)
    return P_inv


def main(args):
    # data loading
    data = np.load('data_npz/Indoor_tx32_c256_csi_DA_trun.npz')
    x_test_norm = data['x_test']
    x_test_norm_para = data['x_test_norm_para']
    x_test = get_csi_denorm(x_test_norm, x_test_norm_para)
    x_test_power_norm_c, x_test_power_norm_para = get_power_norm_c(x_test)
    x_test_power_norm = C2R(x_test_power_norm_c)
    del data
    del x_test
    del x_test_norm
    del x_test_norm_para
    
    FS_data = np.load('data_npz/Indoor_tx32_c256_csi.npz')
    x_FS_norm = FS_data['x_test']
    x_FS_norm_para = FS_data['x_test_norm_para']
    x_FS_origin_c = get_csi_denorm(x_FS_norm, x_FS_norm_para)
    del FS_data
    del x_FS_norm
    del x_FS_norm_para
    
    # model loading
    if args.model_name == 'fddnet':
        channel_input = Input(shape=(32, 32, 2))
        channel_noise = Input(shape=(1,))
        channel_output = fddnet(channel_input, channel_noise, block_num=args.block_num, norm=args.norm,
                                dropout=args.dropout,
                                weight=args.weight, dim=48)
        denoising_model = Model(inputs=[channel_input, channel_noise], outputs=channel_output, name='denoising')
        denoising_model.load_weights(args.model_name_path)

    # paramaters init
    encode_dim = args.encode_dim
    if encode_dim == 1024:
        #
        itr = 10
        rhos_init = 1e-11
        sigma_init = 0.1
        rhos_ratio = 1
        sigma_ratio = 0.6
        set_num = 50
    elif encode_dim == 768:
        #
        itr = 10
        rhos_init = 1e-11
        sigma_init = 0.1
        rhos_ratio = 1
        sigma_ratio = 0.6
        set_num = 50
    elif encode_dim == 512:
        itr = 10
        rhos_init = 1e-11
        sigma_init = 0.1
        rhos_ratio = 1
        sigma_ratio = 0.6
        set_num = 50
    elif encode_dim == 256:
        itr = 10
        rhos_init = 1e-12
        sigma_init = 0.1
        rhos_ratio = 1
        sigma_ratio = 0.3
        set_num = 50
    elif encode_dim == 128:
        itr = 10
        rhos_init = 1e-12
        sigma_init = 1
        rhos_ratio = 1
        sigma_ratio = 0.4
        set_num = 50
    elif encode_dim == 64:
        itr = 10
        rhos_init = 1e-13
        sigma_init = 1
        rhos_ratio = 1
        sigma_ratio = 0.3
        set_num = 5
    elif encode_dim == 32:
        itr = 10
        rhos_init = 1e-13
        sigma_init = 1
        rhos_ratio = 1
        sigma_ratio = 0.3
        set_num = 5
    rhos = [rhos_init * math.pow(rhos_ratio, i) for i in range(itr)]
    sigmas = [sigma_init * math.pow(sigma_ratio, i) for i in range(itr)]

    # results
    mode = args.mode  # 0: power normalization is performed on pnp output results; 1: normalization is performed on each layer of pnp; 2(default): only network output of pnp is normalized
    NMSE_SF = []
    NMSE_SF_PN = []
    NMSE_DA = []
    NMSE_DA_PN = []
    GCS = []
    GCS_PN = []
    NMSE_SF_LIST = []
    NMSE_SF_PN_LIST = []
    NMSE_DA_LIST = []
    NMSE_DA_PN_LIST = []
    GCS_LIST = []
    H_hat = np.zeros((20000, 256, 32)) + 1j * np.zeros((20000, 256, 32))
    
    D_ = np.load('data_npz/random_matrix_0.npz')
    D_ = D_['U']
    D = D_[:, 0:encode_dim, :]
    del D_
    
    _, _, VT = np.linalg.svd(D)
    V11V11T, V12V12T, V11V21T, V12V22T, V21V21T, V22V22T = block_solver(VT, encode_dim)
    del VT

    # SVD and block acceleration #
    for i in range(200):
        t1 = time.time()

        '''
        (2) compression
        '''
        H_real = np.squeeze(x_test_power_norm[i * 100: (i + 1) * 100, :, :, :]).reshape((100, 2048, 1))
        Y = np.matmul(D, H_real)

        '''
        (7) pnp solver
        '''
        # init
        max_set = np.abs(np.matmul(np.transpose(D, (0, 2, 1)), Y))
        index_max = list(np.argsort(max_set, axis=1)[:, ::-1][:, 0:set_num])
        H_ = np.zeros((100, 2048, 1))
        for j in range(100):
            index_max_ = list(index_max[j][:, 0])
            H_[j, index_max_, 0] = np.dot(np.linalg.pinv(D[j, :, index_max_].T), Y[j, :, 0])

        # pnp
        for k in range(itr):
            rho = rhos[k]
            sigma = np.tile(np.expand_dims(np.array(sigmas[k], ), axis=0), 100)
            P_inv = svd_solver(V11V11T, V12V12T, V11V21T, V12V22T, V21V21T, V22V22T, encode_dim, rho)
            Q = 2 * np.matmul(np.transpose(D, (0, 2, 1)), Y) + rho * H_
            Z = np.matmul(P_inv, Q)
            if mode == 1:
                Z, _ = get_power_norm_r(Z)
            Z_ = np.reshape(Z, (100, 32, 32, 2))
            H = denoising_model.predict([Z_, sigma])
            H_ = np.reshape(H, (100, 2048, 1))
            if mode == 1 or mode == 2:
                H_, _ = get_power_norm_r(H_)
        nmse_da_list, nmse_da = cal_nmse_r(H_real, Z)
        NMSE_DA.append(nmse_da)
        NMSE_DA_LIST.append(nmse_da_list)
        
        x_FS_pre_c = DApowernorm_to_HS(Z, x_test_power_norm_para[i * 100: (i + 1) * 100])
        H_hat[i * 100: (i + 1) * 100, :, :] = x_FS_pre_c
        nmse_sf_list, nmse_sf = cal_nmse_c(x_FS_origin_c[i * 100: (i + 1) * 100, :, :], x_FS_pre_c)
        gcs_list, gcs = cal_cosine_similarity_tensor(x_FS_origin_c[i * 100: (i + 1) * 100, :, :], x_FS_pre_c)
        NMSE_SF.append(nmse_sf)
        NMSE_SF_LIST.append(nmse_sf_list)
        GCS.append(gcs)
        GCS_LIST.append(gcs_list)
        #print("{} nmse_da {} ".format(i, nmse_da))

        Z, _ = get_power_norm_r(Z)
        nmse_da_pn_list, nmse_da_pn = cal_nmse_r(H_real, Z)
        NMSE_DA_PN.append(nmse_da_pn)
        NMSE_DA_PN_LIST.append(nmse_da_pn_list)
        
        x_FS_pre_c = DApowernorm_to_HS(Z, x_test_power_norm_para[i * 100: (i + 1) * 100])
        nmse_sf_pn_list, nmse_sf_pn = cal_nmse_c(x_FS_origin_c[i * 100: (i + 1) * 100, :, :], x_FS_pre_c)
        gcs_pn = cal_cosine_similarity_tensor(x_FS_origin_c[i * 100: (i + 1) * 100, :, :], x_FS_pre_c)
        NMSE_SF_PN.append(nmse_sf_pn)
        NMSE_SF_PN_LIST.append(nmse_sf_pn_list)
        GCS_PN.append(gcs_pn)
        
        #print("{} nmse_da_pn {}".format(i, nmse_da_pn))
        t2 = time.time()
        #print("one:{}".format(t2 - t1))
    np.savez('NMSE_Indoor_SVD_POWERNORM_DA_mode{}_{}'.format(mode, encode_dim), NMSE_DA_LIST=NMSE_DA_LIST)
    np.savez('NMSE_Indoor_SVD_POWERNORM_SF_mode{}_{}'.format(mode, encode_dim), NMSE_SF_LIST=NMSE_SF_LIST)
    np.savez('GCS_Indoor_SVD_POWERNORM_SF_mode{}_{}'.format(mode, encode_dim), GCS_LIST=GCS_LIST)
    np.savez('NMSE_Indoor_SVD_POWERNORM_DA_PN_mode{}_{}'.format(mode, encode_dim), NMSE_DA_PN_LIST=NMSE_DA_PN_LIST)
    np.savez('NMSE_Indoor_SVD_POWERNORM_SF_PN_mode{}_{}'.format(mode, encode_dim), NMSE_SF_PN_LIST=NMSE_SF_PN_LIST)
    savemat('Indoor_H_{}.mat'.format(encode_dim), {'H': H_hat})
    print("Eocode_dim {} NMSE_DA{} NMSE_SF{} GCS{} NMSE_DA_PN{} NMSE_SF_PN{} GCS_PN{}".format(encode_dim,
                                                                                              np.mean(NMSE_DA),
                                                                                              np.mean(NMSE_SF),
                                                                                              np.mean(GCS),
                                                                                              np.mean(NMSE_DA_PN),
                                                                                              np.mean(NMSE_SF_PN),
                                                                                              np.mean(GCS_PN)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ''' System parameter'''
    parser.add_argument("-mode", "--mode", default=2, type=int)
    parser.add_argument("-ed", "--encode_dim", default=512, type=int)
    parser.add_argument("-b", "--block_num", default=8, type=int)
    parser.add_argument("-n", "--norm", default='none', type=str)
    parser.add_argument("-w", "--weight", default=1e-6, type=float)
    parser.add_argument("-d", "--dropout", default=0, type=float)
    parser.add_argument("-m", "--model_name", default='fddnet', type=str)
    parser.add_argument("-mp", "--model_name_path",
                        default='./model_zoo/ffdnet_lr0.0001_b8_lnmse_nnone_w1e-06_d0.0_indoor_power_norm.h5', type=str)
    args = parser.parse_args()
    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    main(args)
