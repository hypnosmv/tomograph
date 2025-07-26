import numpy as np
import skimage as ski
import src.utils as utils

class Reconstruction:
    def __init__(self, properties, simulation, apply_filter=False):
        self._init_tomogram(properties, simulation.sinogram)
        if apply_filter:
            self._init_filtered_sinogram(properties, simulation.sinogram)
            self._init_filtered_tomogram(properties)
            self._init_adjusted_tomogram()

    def _init_tomogram(self, properties, sinogram):
        self.tomogram = np.zeros((properties.image.shape[0], properties.image.shape[1]), dtype=np.float32)
        utils.inverse_radon_transform(properties, sinogram, self.tomogram)
        self.raw_tomogram_max = np.max(self.tomogram)
        self.tomogram = utils.normalize_to_0_1(self.tomogram)

    def _init_filtered_sinogram(self, properties, sinogram):
        self.filtered_sinogram = sinogram.copy()

        interval = np.arange(-10, 11)
        if properties.detectors < interval.size:
            interval = interval[(interval.size - properties.detectors) // 2 : (interval.size + properties.detectors) // 2]
        kernel = np.zeros_like(interval, dtype=np.float32)
        odd_indices = interval % 2 != 0
        kernel[interval == 0] = 1
        kernel[odd_indices] = -4 / (np.pi**2 * interval[odd_indices]**2)

        for i in range(properties.scans):
            self.filtered_sinogram[i, :] = np.convolve(self.filtered_sinogram[i, :], kernel, mode='same')

        self.filtered_sinogram = utils.normalize_to_0_1(self.filtered_sinogram)

    def _init_filtered_tomogram(self, properties):
        self.filtered_tomogram = np.zeros((properties.image.shape[0], properties.image.shape[1]), dtype=np.float32)
        utils.inverse_radon_transform(properties, self.filtered_sinogram, self.filtered_tomogram)

        visits = np.ones((properties.image.shape[0], properties.image.shape[1]), dtype=np.int32)

        for i in range(properties.scans):
            for j in range(properties.detectors):
                visits[properties.ray_points[i][j]] += 1

        self.filtered_tomogram /= visits
        self.filtered_tomogram = utils.normalize_to_0_1(self.filtered_tomogram)

    def _init_adjusted_tomogram(self):
        self.adjusted_tomogram = ski.exposure.match_histograms(self.filtered_tomogram, self.tomogram)
