import numpy as np
import skimage as ski

class Properties:
    def __init__(self, image_path, scans, detectors, angular_span):
        self.image = ski.img_as_float32(ski.io.imread(image_path, as_gray=True))
        self.scans = scans
        self.detectors = detectors
        self._phi = np.deg2rad(angular_span, dtype=np.float32)
        self._init_gantry_geometry()
        self._alpha_angles = np.linspace(0, 2*np.pi, scans, endpoint=False, dtype=np.float32)
        self._init_emitter_points()
        self._init_detector_points()
        self._init_ray_points()

    def _init_gantry_geometry(self):
        self.gantry_radius = np.sqrt(np.sum(np.square(self.image.shape)), dtype=np.float32) / 2
        self.gantry_center = np.array(self.image.shape, dtype=np.float32) / 2

    def _init_emitter_points(self):
        cos_sin = np.column_stack((-np.cos(self._alpha_angles), np.sin(self._alpha_angles)))
        self.emitter_points = np.floor(self.gantry_center + self.gantry_radius * cos_sin).astype(np.int32)

    def _init_detector_points(self):
        i = np.arange(self.detectors)
        beta_angles = self._alpha_angles[:, None] + np.pi - self._phi/2 + i * self._phi / (self.detectors - 1)
        cos_sin = np.dstack((-np.cos(beta_angles), np.sin(beta_angles)))
        self.detector_points = np.floor(self.gantry_center + self.gantry_radius * cos_sin).astype(np.int32)

    def _init_ray_points(self):
        self.ray_points = []

        for i in range(self.scans):
            scan_points = []

            for j in range(self.detectors):
                x, y = ski.draw.line_nd(self.emitter_points[i, ::-1], self.detector_points[i, j, ::-1], endpoint=True)
                valid = (y >= 0) & (y < self.image.shape[0]) & (x >= 0) & (x < self.image.shape[1])
                scan_points.append((y[valid], x[valid]))

            self.ray_points.append(scan_points)
