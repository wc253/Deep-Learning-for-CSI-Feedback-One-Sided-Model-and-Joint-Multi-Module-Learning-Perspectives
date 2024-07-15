import numpy as np


def get_csi_norm(x_c):
    x_abs = np.abs(x_c)
    norm_para = np.max(x_abs, axis=(1, 2), keepdims=True)
    x_norm_c = x_c / norm_para / 2
    x_norm_r = np.real(x_norm_c)
    x_norm_r = x_norm_r[:, :, :, np.newaxis]
    x_norm_i = np.imag(x_norm_c)
    x_norm_i = x_norm_i[:, :, :, np.newaxis]
    x_norm = np.concatenate([x_norm_r, x_norm_i], axis=-1)
    x_norm = x_norm + 0.5
    norm_para = np.reshape(norm_para, [-1, ])
    return x_norm, norm_para


def get_csi_norm_3D(x_c):
    x_abs = np.abs(x_c)
    norm_para = np.max(x_abs, axis=(1, 2, 3), keepdims=True)
    x_norm_c = x_c / norm_para / 2
    x_norm_r = np.real(x_norm_c)
    x_norm_r = x_norm_r[:, :, :, :, np.newaxis]
    x_norm_i = np.imag(x_norm_c)
    x_norm_i = x_norm_i[:, :, :, :, np.newaxis]
    x_norm = np.concatenate([x_norm_r, x_norm_i], axis=-1)
    x_norm = x_norm + 0.5
    norm_para = np.reshape(norm_para, [-1, ])
    return x_norm, norm_para


def data_cut(x):
    sample, frequency_domain, antenna = np.size(x)
    tx = 32
    rx = 4
    x = np.reshape(x, newshape=[sample, frequency_domain, rx, tx], order="F")
    x = x.swapaxes(2, 3)
    x_1 = x[:, :, :, 0]
    x_2 = x[:, :, :, 1]
    x_3 = x[:, :, :, 2]
    x_4 = x[:, :, :, 3]
    return x_1, x_2, x_3, x_4


def data_2D_2_3D(x):
    [sample, frequency_domain, antenna] = x.shape
    tx = 32
    rx = 4
    x = np.reshape(x, newshape=[sample, frequency_domain, rx, tx], order="F")
    x = x.swapaxes(2, 3)
    return x

def get_csi_denorm(x_norm, x_norm_para):
    x_norm = x_norm - 0.5
    norm_para = np.reshape(x_norm_para, [-1, 1, 1])
    x_norm_r = x_norm[:, :, :, 0]
    x_norm_i = x_norm[:, :, :, 1]
    x_norm_c = x_norm_r + 1j * x_norm_i
    x_c = x_norm_c * norm_para * 2
    return x_c


def get_csi_denorm_3D(x_norm, x_norm_para):
    x_norm = x_norm - 0.5
    norm_para = np.reshape(x_norm_para, [-1, 1, 1, 1])
    x_norm_r = x_norm[:, :, :, :, 0]
    x_norm_i = x_norm[:, :, :, :, 1]
    x_norm_c = x_norm_r + 1j * x_norm_i
    x_c = x_norm_c * norm_para * 2
    return x_c
    

def get_power_norm(x_c):
    norm_para = np.sqrt(np.mean(np.abs(x_c) ** 2, axis=(1, 2), keepdims=True))
    x_norm_c = x_c / norm_para
    x_norm_r = np.real(x_norm_c)
    x_norm_r = x_norm_r[:, :, :, np.newaxis]
    x_norm_i = np.imag(x_norm_c)
    x_norm_i = x_norm_i[:, :, :, np.newaxis]
    x_norm = np.concatenate([x_norm_r, x_norm_i], axis=-1)
    norm_para = np.reshape(norm_para, [-1, ])
    return x_norm, norm_para


def num2dBm(x):
    return int(np.ceil(10 * np.log(x*1000)/np.log(10)))