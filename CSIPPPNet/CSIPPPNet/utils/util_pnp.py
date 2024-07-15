# -*- coding: utf-8 -*-
import math

import numpy as np
from scipy import interpolate
from utils.util_metric import *


def interp(noisy, r, c, interp='rbf'):
    w = noisy.shape[1]
    interp_noisy = np.zeros((256, 32, 2))
    z_list = []
    for j in range(w):
        z_list.append(noisy[:, j, 0])
    z = np.concatenate(z_list, 0)
    if (interp == 'rbf'):
        f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z, function='gaussian')
        X, Y = np.meshgrid(range(256), range(32))
        z_intp = f(X, Y)
        interp_noisy[:, :, 0] = z_intp.T
    elif (interp == 'spline'):
        tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
        z_intp = interpolate.bisplev(range(256), range(32), tck)
        interp_noisy[:, :, 0] = z_intp
    z_list = []
    for j in range(w):
        z_list.append(noisy[:, j, 1])
    z = np.concatenate(z_list, 0)
    if (interp == 'rbf'):
        f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z, function='gaussian')
        X, Y = np.meshgrid(range(256), range(32))
        z_intp = f(X, Y)
        interp_noisy[:, :, 1] = z_intp.T
    elif (interp == 'spline'):
        tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
        z_intp = interpolate.bisplev(range(256), range(32), tck)
        interp_noisy[:, :, 1] = z_intp
    interp_noisy_complex = np.zeros((256, 32), dtype=complex)
    for i in range(256):
        for j in range(32):
            interp_noisy_complex[i, j] = complex(interp_noisy[i, j, 0], interp_noisy[i, j, 1])
    return interp_noisy_complex


def get_dh_ce(Number_of_pilot, pilot_mode):
    idx = []
    if (Number_of_pilot == 256 and pilot_mode == 0):
        for j in range(0, 32):
            idx = idx + [j + 32 * i for i in range(4 * (j % 8), 256, 32)]
    elif (Number_of_pilot == 256 and pilot_mode == 1):
        for j in range(0, 32):
            idx = idx + [j + 32 * i for i in range(4 * (7 - j % 8), 256, 32)]
    elif (Number_of_pilot == 256 and pilot_mode == 2):
        for j in range(0, 32):
            idx = idx + [j + 32 * i for i in range(8 * (j % 4), 256, 32)]

    elif (Number_of_pilot == 128 and pilot_mode == 0):
        for j in range(0, 32):
            idx = idx + [j + 32 * i for i in range(4 * (j % 16), 256, 64)]
    elif (Number_of_pilot == 128 and pilot_mode == 1):
        for j in range(0, 32):
            idx = idx + [j + 32 * i for i in range(4 * (15 - j % 16), 256, 64)]
    elif (Number_of_pilot == 128 and pilot_mode == 2):
        for j in range(0, 32):
            idx = idx + [j + 32 * i for i in range(8 * (j % 8), 256, 64)]

    elif (Number_of_pilot == 512 and pilot_mode == 0):
        for j in range(0, 32):
            idx = idx + [j + 32 * i for i in range((j % 16), 256, 16)]
    elif (Number_of_pilot == 512 and pilot_mode == 1):
        for j in range(0, 32):
            idx = idx + [j + 32 * i for i in range(2 * (7 - j % 8), 256, 16)]
    elif (Number_of_pilot == 512 and pilot_mode == 2):
        for j in range(0, 32):
            idx = idx + [j + 32 * i for i in range(4 * (j % 4), 256, 16)]

    elif (Number_of_pilot == 1024 and pilot_mode == 0):
        for j in range(0, 32):
            idx = idx + [j + 32 * i for i in range(1 * (j % 8), 256, 8)]
    elif (Number_of_pilot == 1024 and pilot_mode == 1):
        for j in range(0, 32):
            idx = idx + [j + 32 * i for i in range(7 - (j % 8), 256, 8)]
    elif (Number_of_pilot == 1024 and pilot_mode == 2):
        for j in range(0, 32):
            idx = idx + [j + 32 * i for i in range(2 * (j % 4), 256, 8)]
    elif (Number_of_pilot == 1024 and pilot_mode == 3):
        for j in range(0, 32):
            idx = idx + [j + 32 * i for i in range(4 * (j % 2), 256, 8)]

    r = [x // 32 for x in idx]
    c = [x % 32 for x in idx]
    pos = np.zeros((256, 32))
    for j, k in zip(r, c):
        pos[j, k] = 1
    return pos, r, c


def get_dh_ae(Number_of_antenna, antenna_mode):
    idx = []
    if (Number_of_antenna == 8 and antenna_mode == 0):
        for j in range(0, 32, 4):
            idx = idx + [j + 32 * i for i in range(0, 256)]
    elif (Number_of_antenna == 8 and antenna_mode == 1):
        for j in range(2, 32, 4):
            idx = idx + [j + 32 * i for i in range(0, 256)]
    elif (Number_of_antenna == 16 and antenna_mode == 0):
        for j in range(0, 32, 2):
            idx = idx + [j + 32 * i for i in range(0, 256)]
    elif (Number_of_antenna == 16 and antenna_mode == 1):
        for j in range(1, 32, 2):
            idx = idx + [j + 32 * i for i in range(0, 256)]

    r = [x // 32 for x in idx]
    c = [x % 32 for x in idx]
    pos = np.zeros((256, 32))
    for j, k in zip(r, c):
        pos[j, k] = 1
    return pos, r, c


def solve_lmmse(LS_pilot, H, N):
    h = LS_pilot.shape[0]
    w = LS_pilot.shape[1]
    LS_pilot_r = LS_pilot[:, :, 0]
    LS_pilot_i = LS_pilot[:, :, 1]
    LS_pilot_ = LS_pilot_r + 1j * LS_pilot_i
    LS_ = LS_pilot_.reshape(-1, 1)
    H_ = H.reshape(-1, 1)
    RHLS = np.dot(H_, np.matrix(LS_).H)
    RLSLS = np.dot(LS_, np.matrix(LS_).H)
    H_LMMSE = np.dot(np.dot(RHLS, np.linalg.inv(RLSLS + math.sqrt(N / (256 * 32)) * np.eye(h * w))), LS_)
    H_LMMSE = np.array(H_LMMSE.reshape(256, 32))
    return H_LMMSE
