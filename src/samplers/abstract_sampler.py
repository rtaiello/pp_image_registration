""" A collection of samplers"""
from abc import ABC

import numpy as np

from src.common.coordinates import Coordinates2D


class AbstractSampler(ABC):
    def __init__(self, coordinates: Coordinates2D) -> None:
        """
        Initializes the sampler.
        :param coordinates:
        """
        self.coordinates = coordinates

    def f(self, array: np.ndarray, warp: np.ndarray) -> np.ndarray:
        """
        A sampling function, responsible for returning a sampled set of values
        from the given array.
        :param array: input array for sampling
        :param warp: deformation coordinates
        :return: sampled array metric
        """

        if self.coordinates is None:
            raise ValueError("Appropriately defined coordinates not provided.")

        i: np.ndarray = self.coordinates.tensor[0] + warp[0]
        j: np.ndarray = self.coordinates.tensor[1] + warp[1]

        packed_coords = (i.reshape(1, i.size), j.reshape(1, j.size))

        return self.sample(array, np.vstack(packed_coords))

    def sample(self, array: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """
        The sampling function - provided by the specialized samplers.
        """
        pass
