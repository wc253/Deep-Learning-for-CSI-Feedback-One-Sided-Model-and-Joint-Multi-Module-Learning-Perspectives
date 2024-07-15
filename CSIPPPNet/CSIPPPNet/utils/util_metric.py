import numpy as np


def cal_nmse_c(x_c, x_hat_c):
    mse = np.sum(abs(x_c - x_hat_c) ** 2, axis=(1, 2))
    power = np.sum(abs(x_c) ** 2, axis=(1, 2))
    nmse_list = mse / power
    nmse = 10 * np.log10(np.mean(nmse_list))
    return nmse_list, nmse


def cal_nmse_single_c(x_c, x_hat_c):
    mse = np.sum(abs(x_c - x_hat_c) ** 2)
    power = np.sum(abs(x_c) ** 2)
    nmse = 10 * np.log10(mse / power)
    return nmse


def cal_nmse_r(x, x_hat):
    mse = np.sum((x - x_hat) ** 2, axis=(1, 2))
    power = np.sum((x) ** 2, axis=(1, 2))
    nmse_list = mse / power
    nmse = 10 * np.log10(np.mean(nmse_list))
    return nmse_list, nmse


def cal_cosine_similarity_tensor(x, x_hat_c):
    n1 = np.real(np.sqrt(np.sum(np.conj(x) * x, axis=(1, 2))))
    n2 = np.real(np.sqrt(np.sum(np.conj(x_hat_c) * x_hat_c, axis=(1, 2))))
    aa = np.abs(np.sum(np.conj(x_hat_c) * x, axis=(1, 2)))
    rho_list = aa / (n1 * n2)
    rho = np.mean(rho_list)
    return rho_list, rho


def cal_nmse(x, x_hat):
    mse = np.sum(abs(x - x_hat) ** 2)
    power = np.sum(abs(x) ** 2)
    nmse = mse / power
    return nmse


def cal_cosine_similarity(x, x_hat_c):
    rho = 0
    for i in range(32):
        n1 = np.real(np.sqrt(np.sum(np.conj(x[:, i]) * x[:, i])))
        n2 = np.real(np.sqrt(np.sum(np.conj(x_hat_c[:, i]) * x_hat_c[:, i])))
        aa = np.abs(np.sum(np.conj(x_hat_c[:, i]) * x[:, i]))
        rho = rho + aa / (n1 * n2)
    return rho/32.0


def cal_capcity(H, H_feedback, SNR=10):
    Num_sample, M, N = H.shape
    capcity_list = np.zeros(Num_sample)
    snr = pow(10, SNR / 10)

    for i in range(Num_sample):
        capcity = 0
        P_e = np.sum(abs(H[i, :, :] - H_feedback[i, :, :]) ** 2)
        H_hermite = np.matmul(np.squeeze(H_feedback[i, :, :]), np.conj(np.squeeze(H_feedback[i, :, :])).T)
        H_trace = np.trace(H_hermite)
        P_s = N * snr / H_trace
        capcity = np.log2(np.linalg.det(
            np.eye(N) + (snr / ((snr * P_e) / N + 1 / P_s)) * np.matmul(np.conj(np.squeeze(H_feedback[i, :, :])).T,
                                                                        np.squeeze(H_feedback[i, :, :]))))
        capcity_list[i] = capcity
    Capcity = np.sum(capcity_list) / Num_sample
    return Capcity


if __name__ == '__main__':
    H = np.ones((100, 256, 32))
    H_ = np.ones((100, 256, 32))
    rho_, rho = cal_cosine_similarity_tensor(H, H_)
