import numpy as np
import pandas as pd
import os
import sys
BASE_DIR = os.path.dirname(__file__)
print(BASE_DIR)
sys.path.insert(0, BASE_DIR)


def load_train():
    data_dir = os.path.join(BASE_DIR, '../data')
    train_data_fname = os.path.join(data_dir, 'train.csv')
    test_data_fname = os.path.join(data_dir, 'test.csv')
    train_set = pd.read_csv(train_data_fname)
    return train_set


def positions_and_velocities(data, L=None):
    length = L or len(data)
    a = [np.array(data.iloc[i]) for i in range(length)]
    labels = [x[-1] for x in a]
    numbers = [x[1:-2] for x in a]
    mat = np.array(numbers)
    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z = \
        [np.array(mat[:, range(i, mat.shape[1], 7)], dtype=np.float) for i in range(1, 7)]
    return pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, labels



