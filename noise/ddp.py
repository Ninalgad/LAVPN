import numpy as np
from noise.base import NoisingTransform


class DDP(NoisingTransform):
    """
    Diffusion-style implanted noise from:
    `Denoising Pretraining for Semantic Segmentation` (https://arxiv.org/abs/2205.11423)
    """
    def __init__(self, gamma=0.95, sd=1):
        super(DDP, self).__init__("DDP", sd=sd)
        self.gamma = gamma  # [0, 1]

    def add_noise(self, image, mask=None):
        epsilon = np.random.normal(scale=self.sd, size=image.shape)

        noisy = pow(self.gamma, .5) * image.copy() + pow(1 - self.gamma, .5) * epsilon.copy()

        return {"input": noisy.copy(), "target": epsilon.copy(),
                "noise": epsilon.copy(), "dist": epsilon.copy()}
