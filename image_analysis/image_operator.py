from PIL import Image
from PIL import ImageEnhance
import numpy as np


class image_operator_mini:
    def __init__(self, image_filename):
        self.img = Image.open(image_filename)

    def get_brightness_distribution(self):
        pass


"""img = img.rotate(-theta_ratate)
enh = ImageEnhance.Brightness(img)
img = enh.enhance(2)
rx = region_to_show_pix.flatten()
r_img = img.crop((rx[0], rx[3], rx[2], rx[1]))  # reshape(1, -1)
"""
