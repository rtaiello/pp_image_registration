from typing import Tuple

import numpy as np

from src.metrics.abstract_metric import AbstractMetric
from src.models.abstract_model import AbstractModel


class SSD(AbstractMetric):
    """Standard least squares metric"""

    def __init__(self) -> None:
        super().__init__()

    def get_jacobian(
        self, model: AbstractModel, warped_image: np.ndarray, p=None
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            A jacobain matrix. (m x n)
                | where: m = number of image pixels,
                |        p = number of parameters.
        """
        grad = np.gradient(warped_image)
        grad_norm = np.linalg.norm(np.array(grad), axis=0).flatten()

        dIx = grad[1].flatten()
        dIy = grad[0].flatten()

        dPx, dPy = model.jacobian(p)

        J = np.zeros_like(dPx)
        for index in range(0, dPx.shape[1]):
            J[:, index] = dPx[:, index] * dIx + dPy[:, index] * dIy
        return J, grad_norm

    def error(self, warped_image: np.ndarray, template_image: np.ndarray) -> np.ndarray:
        """
        Evaluates the residual metric.

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
        result: np.ndarray = warped_image.flatten() - template_image.flatten()
        return result
