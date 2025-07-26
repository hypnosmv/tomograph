import numpy as np
import src.utils as utils

class Simulation:
    def __init__(self, properties):
        self._init_sinogram(properties)

    def _init_sinogram(self, properties):
        self.sinogram = np.zeros((properties.scans, properties.detectors), dtype=np.float32)
        utils.radon_transform(properties, self.sinogram)
        self.raw_sinogram_max = np.max(self.sinogram)
        self.sinogram = utils.normalize_to_0_1(self.sinogram)
