import numpy as np


def cal_downlink_R_sum(V_c, H_dl_RB_1, H_dl_RB_2, P_matrix, noise_Power):
    H_dl_RB_1 = H_dl_RB_1[...,0]+1j*H_dl_RB_1[...,1]
    H_dl_RB_2 = H_dl_RB_2[..., 0] + 1j * H_dl_RB_2[..., 1]

    H_dl_1 = H_dl_RB_1[..., np.newaxis]  # 100000*52*32*4*1
    H_dl_2 = H_dl_RB_2[..., np.newaxis]  # 100000*52*32*4*1
    H_dl = np.concatenate([H_dl_1, H_dl_2], axis=-1)  # 100000*52*32*4*2
    for k_index in range(2):
        H_RB = np.squeeze(H_dl[..., k_index])  # 100000*52*32*4
        for i in range(13):  # 计算13个子带的和
            for j in range(4):  # 计算每个子带中4个RB的和
                P = P_matrix[:, i, :]  # 100000*2
                H = np.transpose(np.squeeze(H_RB[:, 4 * i + j, :, :]), [0, 2, 1])  # 100000*4*32
                F = np.squeeze(V_c[:, :, i, k_index])[..., np.newaxis]  # 100000*32*1
                norm_F = np.sqrt(np.sum(np.abs(F)**2, axis=1, keepdims=True))  # 10000*1
                norm_F = norm_F + 1j * 0 * norm_F
                F = np.divide(F, norm_F)
                HF = np.matmul(H, F)  # 100000*4*1
                HF_norm = np.sqrt(np.sum(np.square(np.abs(HF)), axis=1, keepdims=True))
                HF_norm_c = HF_norm + 1j * np.zeros(shape=HF_norm.shape)
                W = HF / HF_norm_c
                W_H = np.conj(np.transpose(W, [0, 2, 1]))
                for kk in range(2):  # 计算k_index用户在每个RB上的R
                    V = np.squeeze(V_c[:, :, i, kk])[..., np.newaxis]
                    norm_V = np.sqrt(
                        np.sum(np.abs(V) ** 2, axis=1,
                                      keepdims=True))  # 10000*1
                    norm_V = norm_V+1j * 0 * norm_V
                    V = np.divide(V, norm_V)
                    H_V = np.matmul(H, V)
                    WH_H_V = np.matmul(W_H, H_V)
                    WHVVHW = np.square(np.abs(WH_H_V))
                    norm = np.squeeze(WHVVHW) * P[:, kk]
                    if kk == k_index:
                        nom = norm
                    if kk == 0:
                        noise = noise_Power + 1j * 0
                        eye_matrix = np.eye(4, 4)[np.newaxis,...]
                        noise_matrix = (noise * eye_matrix).repeat(W_H.shape[0],axis=0)
                        WZZ = np.matmul(W_H, noise_matrix)
                        WZZW = np.squeeze(np.matmul(WZZ, W))
                        nom_denom = norm + WZZW
                    else:
                        nom_denom = nom_denom + norm
                denom = nom_denom - nom
                rate = np.log(1 + np.abs(np.divide(nom, denom))) / np.log(2.0)
                if i == 0:
                    if j == 0:
                        R_ = rate
                    elif 0 < j < 3:
                        R_ = R_ + rate
                    else:
                        R_ = R_ + rate
                        R = R_[..., np.newaxis]
                else:
                    if j == 0:
                        R_ = rate
                    elif 0 < j < 3:
                        R_ = R_ + rate
                    else:
                        R_ = R_ + rate
                        R_ = R_[..., np.newaxis]
                        R = np.concatenate([R, R_], axis=-1)
        if k_index == 0:
            R_sum = np.mean(R, axis=-1) / 4
            R_1 = R_sum
        else:
            R_2 = np.mean(R, axis=-1) / 4
            R_sum = R_sum + R_2

    R_sum = np.mean(R_sum)
    A = np.concatenate([R_1[..., np.newaxis], R_2[..., np.newaxis]], axis=-1)   # just for check!
    R_1 = np.mean(R_1)
    R_2 = np.mean(R_2)
    return R_1, R_2, R_sum

def cal_nmse_2D(x_c, x_hat_c):
    mse = np.sum(abs(x_c - x_hat_c) ** 2, axis=(1, 2))
    power = np.sum(abs(x_c) ** 2, axis=(1, 2))
    nmse = 10 * np.log10(np.mean(mse / power))
    return nmse
    
def cal_SGCS(x, x_hat_c):
    aa = np.abs(np.sum(np.conj(x_hat_c) * x, axis=1))  # 100000*13
    n1 = np.real(np.sqrt(np.sum(np.conj(x) * x, axis=1)))  # 100000*13
    n2 = np.real(np.sqrt(np.sum(np.conj(x_hat_c) * x_hat_c, axis=1)))  # 100000*13
    n = n1 * n2  # 100000*13
    GCS = aa/n
    SGCS = np.square(GCS)
    SGCS = np.mean(SGCS)
    return SGCS