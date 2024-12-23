import numpy as np
from noise.base import NoisingTransform
from noise.utils import *


def sigmoid(z):
    return np.exp(z)/(1 + np.exp(z))


class LAVPN(NoisingTransform):
    def __init__(self, sd=1, phin0=1, phin1=2):
        super(LAVPN, self).__init__("LAVPN", sd=sd)
        self.phin0, self.phin1 = phin0, phin1

    def add_noise(self, image, mask):
        mask = resample_mask(one_hot2dist(mask))

        epsilon = np.random.normal(scale=self.sd, size=image.shape)

        phi = (self.phin1 - self.phin0) * mask + self.phin0
        phi /= 2.0*np.pi

        z = np.cos(phi) * image + np.sin(phi) * epsilon
        v = np.cos(phi) * epsilon - np.sin(phi) * image

        return {"input": z.copy(), "target": v.copy(),
                "noise": epsilon.copy(), "dist": phi.copy()}
