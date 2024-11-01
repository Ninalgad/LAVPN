from scipy import ndimage
import numpy as np

SCALING = {"1ADC_ss": (1350.2495, 428.13467), "2Z_ADC": (0.34669298, 2.487756)}


def preprocess(img, ms, target_size=None):
    if target_size is not None:
        eps = 1e-10
        sx = target_size / img.shape[1] + eps
        sy = target_size / img.shape[2] + eps

        mask = img != 0
        img = ndimage.zoom(img, (1, sx, sy), cval=0.0)
        mask = ndimage.zoom(mask.astype('uint8'), (1, sx, sy), cval=0.0)
        img = img * mask.astype(img.dtype)

        img = img[:, :target_size, :target_size]
        mask = mask[:, :target_size, :target_size]

        assert (target_size, target_size) == img.shape[-2:], img.shape
        assert (target_size, target_size) == mask.shape[-2:], mask.shape

    m, s = ms
    idx = img != 0
    img[idx] = (img[idx] - m) / pow(s, 1.2)
    idx = np.logical_not(idx)
    img[idx] = -6
    img = np.clip(img, -6, 6)

    img = np.expand_dims(img, -1)
    return img


def create_input_array(zadc, ss_adc, target_size=None, channels_first=False):
    ss_adc = preprocess(ss_adc, SCALING['1ADC_ss'], target_size)
    zadc = preprocess(zadc, SCALING['2Z_ADC'], target_size)

    img = np.concatenate([ss_adc, zadc, zadc], axis=-1)
    if channels_first:
        img = np.transpose(img, (0, 3, 1, 2))

    return img
