import os
import base64
import tempfile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import display, HTML

def normalize_to_0_1(array):
    return ((array - np.min(array)) / (np.max(array) - np.min(array))).astype(np.float32)

def display_image(array):
    if array.dtype != np.float32 or array.min() < 0.0 or array.max() > 1.0:
        raise ValueError(f'Array type={array.dtype}, min={array.min()}, max={array.max()}, expected float32 in [0.0, 1.0]')
    display(Image.fromarray((array * 255.0).astype(np.uint8)))

def radon_transform(properties, array, i_begin=0, i_end=None):
    if i_end is None: i_end = properties.scans

    for i in range(i_begin, i_end):
        for j in range(properties.detectors):
            ray_length = np.linalg.norm(properties.emitter_points[i] - properties.detector_points[i, j])
            pixel_values = properties.image[properties.ray_points[i][j]]
            if ray_length > 0:
                array[i, j] = np.sum(pixel_values) / ray_length

def inverse_radon_transform(properties, sinogram, array, i_begin=0, i_end=None):
    if i_end is None: i_end = properties.scans

    for i in range(i_begin, i_end):
        for j in range(properties.detectors):
            array[properties.ray_points[i][j]] += sinogram[i, j]

def create_animation(properties, shape, raw_max, update_array=lambda sb, se, wa: None, scan_step=40, interval=800):
    dpi = 100
    fig = plt.figure(figsize=(shape[1] / dpi, shape[0] / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    working_array = np.zeros(shape, dtype=np.float32)
    working_image = ax.imshow(working_array, cmap='gray', vmin=0, vmax=raw_max)

    frame_count = properties.scans // scan_step + 1 + (1 if properties.scans % scan_step > 0 else 0)

    def update(frame):
        scan_begin = (frame - 1) * scan_step
        scan_end = min(frame * scan_step, properties.scans)

        if frame > 0:
            update_array(scan_begin, scan_end, working_array)
            working_image.set_array(working_array)

        return [working_image]

    tomogram_animation = FuncAnimation(fig, update, frames=frame_count, interval=interval, blit=True)

    plt.close(fig)

    gif_path = tempfile.NamedTemporaryFile(suffix=".gif", delete=False).name

    try:
        tomogram_animation.save(gif_path, writer=PillowWriter(fps=1000/interval))
        gif_data = open(gif_path, 'rb').read()
    finally:
        os.remove(gif_path)

    gif_base64 = base64.b64encode(gif_data).decode('utf-8')

    return HTML(f'<img src="data:image/gif;base64,{gif_base64}" alt="Animation" />')
