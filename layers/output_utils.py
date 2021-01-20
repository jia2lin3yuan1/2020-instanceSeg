""" Contains functions used to sanitize and prepare the output of Yolact. """

import cv2
import numpy as np
from data import MEANS, STD

def undo_image_transformation(bkbone_cfg, img, w=None, h=None):
    """
    Takes a transformed image tensor and returns a numpy ndarray that is untransformed.
    Arguments w and h are the original height and width of the image.
    """
    img_numpy = img.permute(1, 2, 0).cpu().numpy()
    img_numpy = img_numpy[:, :, (2, 1, 0)] # To BRG

    if bkbone_cfg.transform.normalize:
        img_numpy = (img_numpy * np.array(STD) + np.array(MEANS)) / 255.0
    elif bkbone_cfg.transform.subtract_means:
        img_numpy = (img_numpy / 255.0 + np.array(MEANS) / 255.0).astype(np.float32)

    img_numpy = img_numpy[:, :, (2, 1, 0)] # To RGB
    img_numpy = np.clip(img_numpy, 0, 1)

    if w is not None and h is not None:
        return cv2.resize(img_numpy, (w,h))
    else:
        return img_numpy
