from abc import ABC, abstractmethod

import numpy as np

from src.models.abstract_model import AbstractModel


class AbstractMetric(ABC):


    @abstractmethod
    def error(self, warped_image: np.ndarray, template_image: np.ndarray) -> np.ndarray:
        """
        Evaluates the metric.

        Parameters
        ----------
        warped_image: nd-array
            Input image after warping.
        template_image: nd-array
            Template image.

        Returns
        -------
        error: nd-array
           Metric evaluated over all image coordinates.
        """

        pass

    @abstractmethod
    def get_jacobian(
        self, model: AbstractModel, warped_image: np.ndarray, p=None
    ) -> np.ndarray:
        """
        Computes the jacobian dP/dE.

        Parameters
        ----------
        model: deformation model
            A particular deformation model.
        warped_image: nd-array
            Input image after warping.
        p : optional list
            Current warp parameters

        Returns
        -------
        jacobian: nd-array
           A derivative of model parameters with respect to the metric.
        """
        pass

    def __str__(self) -> str:
        return "Metric: {0} \n {1}".format(self.METRIC, self.DESCRIPTION)
