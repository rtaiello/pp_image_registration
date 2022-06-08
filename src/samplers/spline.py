import numpy as np
import scipy.ndimage as nd

from src.samplers.abstract_sampler import AbstractSampler


class Spline(AbstractSampler):

    METHOD = "nd-image spline sampler (SR)"

    DESCRIPTION = """
        Refer to the documentation for the ndimage map_coordinates function.

        http://docs.scipy.org/doc/scipy/reference/generated/
            scipy.ndimage.interpolation.map_coordinates.html
        """

    def __init__(self, coordinates):
        AbstractSampler.__init__(self, coordinates)

    def f(self, array: np.ndarray, warp: np.ndarray) -> np.ndarray:
        """
        A sampling function, responsible for returning a sampled set of values
        from the given array.

        Parameters
        ----------
        array: nd-array
            Input array for sampling.
        warp: nd-array
            Deformation coordinates.

        Returns
        -------
        sample: nd-array
           Sampled array metric.
        """

        if self.coordinates is None:
            raise ValueError("Appropriately defined coordinates not provided.")

        return nd.map_coordinates(array, warp, order=2, mode="nearest").flatten()
