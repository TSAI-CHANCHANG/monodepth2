from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset

class SevenDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SevenDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[585, 0, 320, 0],
                           [0, 585, 240, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (1242,375)

    def check_depth(self):
        return True

    def get_color(self, folder, frame_index, side, do_flip):
        f_str = "frame-{:06d}.color{}".format(frame_index, ".png")
        image_path = os.path.join(self.data_path, folder, "rgb", f_str)
        color = self.loader(image_path)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "frame-{:06d}.depth{}".format(frame_index, ".png")
        image_path = os.path.join(self.data_path, folder, "depth", f_str)
        depth = self.loader(image_path)
        if do_flip:
            depth = depth.transpose(pil.FLIP_LEFT_RIGHT)
        
        return depth
    