import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance


def one_hot2dist(seg: np.ndarray, resolution=(1, 1),
                 dtype='float32') -> np.ndarray:
    num_channels = seg.shape[-1]

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(num_channels):
        posmask = seg[:, :, k].astype('bool')

        if posmask.any():
            negmask = ~posmask
            res[:, :, k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
    return res


def resample_mask(dist):

    c = np.random.beta(1, 1)  # [0,1] creat distance
    c = (c - 0.5) * 2  # [-1.1]
    m = np.min(dist)
    if c > 0:
        lam = c * m / 2  # λl = -1/2|min(dis_array)|
    else:
        lam = c * m
    mask = (dist < lam).astype('float32')  # creat M
    return mask
