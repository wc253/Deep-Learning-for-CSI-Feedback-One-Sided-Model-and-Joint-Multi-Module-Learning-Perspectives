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


def get_csi_denorm(x_norm, x_norm_para):
    x_norm = x_norm - 0.5
    norm_para = np.reshape(x_norm_para, [-1, 1, 1])
    x_norm_r = x_norm[:, :, :, 0]
    x_norm_i = x_norm[:, :, :, 1]
    x_norm_c = x_norm_r + 1j * x_norm_i
    x_c = x_norm_c * norm_para * 2
    return x_c


def R2C(H):
    x_r = H[:, :, :, 0]
    x_i = H[:, :, :, 1]
    x_c = x_r + 1j * x_i
    return x_c


def C2R(H):
    x_r = np.real(H)
    x_r = x_r[:, :, :, np.newaxis]
    x_i = np.imag(H)
    x_i = x_i[:, :, :, np.newaxis]
    x = np.concatenate([x_r, x_i], axis=-1)
    return x


def r2C(H):
    H = np.reshape(H, (32, 32, 2))
    x_r = H[:, :, 0]
    x_i = H[:, :, 1]
    x_c = x_r + 1j * x_i
    return x_c


def C2r(H):
    x_r = np.real(H)
    x_r = x_r[:, :, np.newaxis]
    x_i = np.imag(H)
    x_i = x_i[:, :, np.newaxis]
    x = np.concatenate([x_r, x_i], axis=-1)
    return x


def get_csi_denorm_pnp(x_norm, x_norm_para):
    x_norm = np.reshape(x_norm, (100, 32, 32, 2))
    norm_para = np.reshape(x_norm_para, [-1, 1, 1])
    x_norm_r = x_norm[:, :, :, 0]
    x_norm_i = x_norm[:, :, :, 1]
    x_norm_c = x_norm_r + 1j * x_norm_i
    x_c = x_norm_c * norm_para
    return x_c


def DAnorm_to_HS(x_norm, x_norm_para):
    x_trun_c = get_csi_denorm_pnp(x_norm, x_norm_para)
    shape = np.shape(x_trun_c)
    x_c = np.concatenate([x_trun_c, np.zeros(shape=[shape[0], 256 - shape[1], shape[2]])], axis=1)
    x_DS = np.fft.ifft(x_c, axis=2)
    x_FS = np.fft.fft(x_DS, axis=1)
    return x_FS


def get_power_norm_c(x_c):
    power_norm_para = np.sqrt(np.sum(np.abs(x_c) ** 2, axis=(1, 2), keepdims=True))
    x_power_norm_c = x_c / power_norm_para
    power_norm_para = np.reshape(power_norm_para, [-1, ])
    return x_power_norm_c, power_norm_para


def get_power_norm_r(x):
    power_norm_para = np.sqrt(np.sum(x ** 2, axis=(1, 2), keepdims=True))
    x_power_norm = x / power_norm_para
    power_norm_para = np.reshape(power_norm_para, [-1, ])
    return x_power_norm, power_norm_para


def get_power_norm_single(x_c):
    power_norm_para = np.sqrt(np.sum(np.abs(x_c) ** 2))
    x_power_norm_c = x_c / power_norm_para
    power_norm_para = np.reshape(power_norm_para, [-1, ])
    return x_power_norm_c, power_norm_para


def DApowernorm_to_HS(x_norm, x_norm_para):
    x_trun_c = get_power_denorm(x_norm, x_norm_para)
    shape = np.shape(x_trun_c)
    x_c = np.concatenate([x_trun_c, np.zeros(shape=[shape[0], 256 - shape[1], shape[2]])], axis=1)
    x_DS = np.fft.ifft(x_c, axis=2)
    x_FS = np.fft.fft(x_DS, axis=1)
    return x_FS


def get_power_denorm(x_norm, x_norm_para):
    x_norm = np.reshape(x_norm, (100, 32, 32, 2))
    norm_para = np.reshape(x_norm_para, [-1, 1, 1])
    x_norm_r = x_norm[:, :, :, 0]
    x_norm_i = x_norm[:, :, :, 1]
    x_norm_c = x_norm_r + 1j * x_norm_i
    x_c = x_norm_c * norm_para
    return x_c


def DApowernorm_to_HS_single(x_norm, x_norm_para):
    x_trun_c = get_power_denorm_signle(x_norm, x_norm_para)
    shape = np.shape(x_trun_c)
    x_c = np.concatenate([x_trun_c, np.zeros(shape=[shape[0], 256 - shape[1], shape[2]])], axis=1)
    x_DS = np.fft.ifft(x_c, axis=2)
    x_FS = np.fft.fft(x_DS, axis=1)
    return x_FS


def get_power_denorm_signle(x_norm, x_norm_para):
    x_norm = np.reshape(x_norm, (1, 32, 32, 2))
    norm_para = np.reshape(x_norm_para, [-1, 1, 1])
    x_norm_r = x_norm[:, :, :, 0]
    x_norm_i = x_norm[:, :, :, 1]
    x_norm_c = x_norm_r + 1j * x_norm_i
    x_c = x_norm_c * norm_para
    return x_c


def DA_to_HS(x_norm):
    x_trun_c = x_norm
    shape = np.shape(x_trun_c)
    x_c = np.concatenate([x_trun_c, np.zeros(shape=[shape[0], 256 - shape[1], shape[2]])], axis=1)
    x_DS = np.fft.ifft(x_c, axis=2)
    x_FS = np.fft.fft(x_DS, axis=1)
    return x_FS

def HS_to_DA(x_norm):
    x_DS = np.fft.ifft(x_norm, axis=1)
    x_AD = np.fft.fft(x_DS, axis=2)
    x_AD_trun = x_AD[:, 0:32, :]
    return x_AD_trun

