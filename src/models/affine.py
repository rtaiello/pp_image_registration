from typing import Optional, Tuple

import numpy as np

from src.common.coordinates import Coordinates2D
from src.models.abstract_model import AbstractModel


class Affine(AbstractModel):
    """
    Applies the affine coordinate transformation. Follows the derivations
    shown in:

    S. Baker and I. Matthews. 2004. Lucas-Kanade 20 Years On: A
    Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
    """

    def __init__(self, coordinates: Coordinates2D) -> None:
        AbstractModel.__init__(self, coordinates)

    @property
    def identity(self):
        """
        Returns the identity affine transformation.
        :return:
        """
        return np.zeros(6)

    @staticmethod
    def scale(p: np.ndarray, factor: float) -> np.ndarray:
        """
        Scales an affine transformation by a factor.
        :param p: model parameters
        :param factor: scaling factor
        :return: scaled model parameters
        """
        pHat = p.copy()
        pHat[4:] *= factor
        return pHat

    def fit(
        self, p0: np.ndarray, p1: np.ndarray, lmatrix: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimates the best fit parameters that define a warp field, which
        deforms feature points p0 to p1.
        :param p0: image features (points)
        :param p1: template features (points)
        :param lmatrix:
        :return: parmaters: model parameters, error: sum of RMS error between p1 and aligned p0
        """

        # Solve: H*X = Y
        # ---------------------
        #          H = Y*inv(X)

        X = np.ones((3, len(p0)))
        X[0:2, :] = p0.T

        Y = np.ones((3, len(p0)))
        Y[0:2, :] = p1.T

        H = np.dot(Y, np.linalg.pinv(X))

        parameters: np.ndarray = np.array(
            [H[0, 0] - 1.0, H[1, 0], H[0, 1], H[1, 1] - 1.0, H[0, 2], H[1, 2]]
        )

        projP0 = np.dot(H, X)[0:2, :].T

        error = np.sqrt(
            (projP0[:, 0] - p1[:, 0]) ** 2 + (projP0[:, 1] - p1[:, 1]) ** 2
        ).sum()

        return parameters, error

    def transform(self, p: np.ndarray) -> np.ndarray:
        """
        An affine transformation of coordinates.
        :param p: model parameters
        :return: deformation coordinates
        """

        T = np.array([[p[0] + 1.0, p[2], p[4]], [p[1], p[3] + 1.0, p[5]], [0, 0, 1]])

        displacement = (
            np.dot(np.linalg.inv(T), self.coordinates.homogenous)
            - self.coordinates.homogenous
        )

        shape = self.coordinates.tensor[0].shape

        return np.array(
            [displacement[1].reshape(shape), displacement[0].reshape(shape)]
        )

    def jacobian(self, p: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the Jacobian of the affine transformation.
        :param p: model parameters
        :return: derivative of the affine transformation
        """

        dx = np.zeros((self.coordinates.tensor[0].size, 6))
        dy = np.zeros((self.coordinates.tensor[0].size, 6))

        dx[:, 0] = self.coordinates.tensor[1].flatten()
        dx[:, 2] = self.coordinates.tensor[0].flatten()
        dx[:, 4] = 1.0

        dy[:, 1] = self.coordinates.tensor[1].flatten()
        dy[:, 3] = self.coordinates.tensor[0].flatten()
        dy[:, 5] = 1.0

        return dx, dy
