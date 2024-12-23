from abc import ABC, abstractmethod


class NoisingTransform(ABC):
    def __init__(self, name, sd=1):
        self.name = name
        self.sd = sd

    def apply(self, image, mask):
        background = image == -6
        noise_transformed = self.add_noise(image, mask)
        noise_transformed["input"][background] = -6
        return noise_transformed

    @abstractmethod
    def add_noise(self, image, mask):
        raise NotImplementedError("Please Implement this method")
