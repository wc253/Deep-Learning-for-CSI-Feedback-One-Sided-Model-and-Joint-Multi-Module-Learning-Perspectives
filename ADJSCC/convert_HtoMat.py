import numpy as np
import os
import scipy.io as sio


data_path = 'H:\dataset_38.901\indoor/Indoor_tx32_c256_H.npz'
data = np.load(data_path)

H_test = data['x_test']
sio.savemat('data_npz/H.mat', {'H_test': H_test})
