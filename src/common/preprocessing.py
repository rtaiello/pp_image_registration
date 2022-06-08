import nibabel as nib
import numpy as np
import scipy.ndimage as nd
from PIL import Image

from src.registers.abstract_register import RegisterData


def linear(image_path, template_path):
    template: np.ndarray = np.array(Image.open(template_path).convert("L"))
    image: np.ndarray = np.array(Image.open(image_path).convert("L"))
    return image, template


def non_linear(image_path, template_path):
    """

    :param image_path: path to the image
    :param template_path:   path to the template
    :return:
    """
    template: np.ndarray = np.nan_to_num(nib.load(template_path).get_fdata()[:, :])
    template[template < 1e-3] = 0.0
    image: np.ndarray = np.nan_to_num(nib.load(image_path).get_fdata()[:, :])
    image[image < 1e-3] = 0.0

    return image, template


def non_linear_supplementary(image_path, template_path):
    template: np.ndarray = np.array(Image.open(template_path).convert("L"))
    image = np.array(Image.open(image_path).convert("L"))
    template = RegisterData._smooth(template, 0.5)
    image = RegisterData._smooth(image, 0.5)
    return image, template
