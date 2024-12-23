import numpy as np
from noise.base import NoisingTransform
from noise.utils import *


class LADP(NoisingTransform):
    """
    Label-aware diffusion style nose implantation
    """
    def __init__(self, sd=0.8, s0=.8, s1=1.5, gamma=0.95):
        super(LADP, self).__init__('LADP', sd)
        self.s0 = s0
        self.s1 = s1
        self.gamma = gamma

    def add_noise(self, image, mask):
        dist = one_hot2dist(mask)
        mask = resample_mask(dist)

        s0, s1 = self.s0, self.s1
        y = np.random.normal(scale=self.sd, size=image.shape)

        z = y.copy()
        z = z * (1 - mask) * s0 + z * mask * s1
        noisy_image = pow(self.gamma, .5) * image.copy() + pow(1 - self.gamma, .5) * z

        return {"input": noisy_image, "target": y,
                "noise": y, "dist": dist}
