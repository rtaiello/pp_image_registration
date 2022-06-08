""" A collection of deformation models. """

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from src.common.coordinates import Coordinates2D


class AbstractModel(ABC):
    """Abstract model class."""

    def __init__(self, coordinates: Coordinates2D) -> None:
        """
        Initializes the model.
        :param coordinates:
        """
        self.coordinates = coordinates

    def fit(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """
        Estimates the best fit parameters that define a warp field, which
        deforms feature points p0 to p1.
        :param p0: image features (points)
        :param p1: template features (points)
        :return: Model parameters
        """
        pass

    @staticmethod
    def scale(p: np.ndarray, factor: float) -> np.ndarray:
        """
        Scales a transformation by a factor.
        :param p: model parameters
        :param factor: scaling factor
        :return: scaled model parameters
        """

        pass

    def estimate(self, warp: np.ndarray) -> np.ndarray:
        """
        Estimates the best fit parameters that define a warp field.

        :param warp: deformation field
        :return: model parameters
        """
        pass

    def warp(self, parameters: np.ndarray) -> np.ndarray:
        """
        Computes the warp field given model parameters.
        :param parameters: model parameters
        :return: deformation field
        """

        displacement = self.transform(parameters)

        # Approximation of the inverse (samplers work on inverse warps).
        return self.coordinates.tensor + displacement

    @abstractmethod
    def transform(self, parameters: np.ndarray) -> np.ndarray:
        """
        A geometric transformation of coordinates.
        :param parameters: model parameters
        :return: deformation coordinates
        """
        pass

    @abstractmethod
    def jacobian(self, p=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the derivative of deformation model with respect to the
        coordinates.
        """
        pass
