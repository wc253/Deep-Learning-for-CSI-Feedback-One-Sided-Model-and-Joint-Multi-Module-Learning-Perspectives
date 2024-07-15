import numpy as np


def cal_nmse(x_c, x_hat_c):
    mse = np.sum(abs(x_c - x_hat_c) ** 2, axis=(1, 2))
    power = np.sum(abs(x_c) ** 2, axis=(1, 2))
    nmse = 10 * np.log10(np.mean(mse / power))
    return nmse


def cal_cosine_similarity(x, x_hat_c):
    n1 = np.real(np.sqrt(np.sum(np.conj(x) * x, axis=2)))
    n2 = np.real(np.sqrt(np.sum(np.conj(x_hat_c) * x_hat_c, axis=2)))
    aa = np.abs(np.sum(np.conj(x_hat_c) * x, axis=2))
    rho = np.mean(aa / (n1 * n2))
    return rho
