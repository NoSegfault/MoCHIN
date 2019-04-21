# https://github.com/ysyushi/prep/blob/master/src/GradDescent.py

import numpy as np


# y: 2-D array
def simplex_project(y, infinitesimal):
    # 1-D vector version
    D = len(y)
    u = np.sort(y)[::-1]
    x_tmp = (1. - np.cumsum(u)) / np.arange(1, D+1)
    lmd = x_tmp[np.sum(u + x_tmp > 0) - 1]
    return np.maximum(y + lmd, 0)

    '''
    n, d = y.shape
    x = np.fliplr(np.sort(y, axis=1))
    x_tmp = np.dot((np.cumsum(x, axis=1) + (d * infinitesimal - 1.)), np.diagflat(1. / np.arange(1, d + 1)))
    lmd = x_tmp[np.arange(n), np.sum(x > x_tmp, axis=1) - 1]
    return np.maximum(y - lmd[:, np.newaxis], 0) + infinitesimal
    '''