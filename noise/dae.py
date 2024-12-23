import numpy as np
from noise.base import NoisingTransform


class DAE(NoisingTransform):
    """
    Constant implanted noise predicting the clean image from:
    `Extracting and Composing Robust Features with Denoising Autoencoders`
    (cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)
    """
    def __init__(self, sd=1):
        super(DAE, self).__init__("DAE", sd=sd)

    def add_noise(self, image, mask=None):
        epsilon = np.random.normal(scale=self.sd, size=image.shape)

        noisy = image.copy() + epsilon.copy()

        return {"input": noisy.copy(), "target": image.copy(),
                "noise": epsilon.copy(), "dist": epsilon.copy()}


class NLDAE(NoisingTransform):
    """
    Constant implanted noise predicting the noise:
    `Noise Learning Based Denoising Autoencoder`
    (https://arxiv.org/pdf/2101.07937)
    """
    def __init__(self, sd=1):
        super(NLDAE, self).__init__("NLDAE", sd=sd)

    def add_noise(self, image, mask=None):
        epsilon = np.random.normal(scale=self.sd, size=image.shape)

        noisy = image.copy() + epsilon.copy()

        return {"input": noisy.copy(), "target": image.copy(),
                "noise": epsilon.copy(), "dist": epsilon.copy()}
