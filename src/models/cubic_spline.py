from typing import Optional, Tuple

import hydra
import numpy as np
import scipy.signal as signal

from src.models.abstract_model import AbstractModel


class CubicSpline(AbstractModel):
    """
    Applies a cubic_spline deformation model, as described in:
    Kybic, J. and Unser, M. (2003). Fast parametric elastic image
    registration. IEEE Transactions on Image Processing, 12(11), 1427-1442.
    """

    def __init__(self, coordinates):
        """
        Initializes the model.
        :param coordinates:
        """
        AbstractModel.__init__(self, coordinates)
        self.__basis()

    @property
    def identity(self):
        """
        Returns the identity warp field.
        :return:
        """
        return np.zeros(self.basis.shape[1] * 2)

    @property
    def number_parameters(self) -> int:
        """Returns the number of parameters."""
        return self.basis.shape[1]

    def __basis(self, order=3, divisions=5, spacing: Optional[int] = 5):

        """
        Computes the spline tensor product and stores the products, as basis
        vectors.
        Parameters
        ----------
        order: int
            B-spline order, optional.
        divisions: int, optional.
            Number of spline knots.
        """

        shape = self.coordinates.tensor[0].shape
        grid = self.coordinates.tensor
        spacing = shape[1] // divisions if spacing is None else spacing
        x_knots = shape[1] // spacing
        y_knots = shape[0] // spacing

        hydra.utils.log.info(
            f"Shape of the image: {self.coordinates.tensor[0].shape} - Number of control points: x {x_knots} - y {y_knots} - Spacing x {spacing} - Spacing y {spacing}"
        )

        qx = np.zeros((grid[0].size, x_knots))
        qy = np.zeros((grid[0].size, y_knots))

        for index in range(0, x_knots):
            # Compute the basis vectors for the x-coordinate.
            bx = signal.bspline(grid[1] / spacing - index, order)
            qx[:, index] = bx.flatten()

        for index in range(0, y_knots):
            # Compute the basis vectors for the y-coordinate.
            by = signal.bspline(grid[0] / spacing - index, order)
            qy[:, index] = by.flatten()

        basis = []
        for j in range(0, x_knots):
            for k in range(0, y_knots):
                basis.append(qx[:, j] * qy[:, k])
        self.basis = np.array(basis).T

    def estimate(self, warp: np.ndarray) -> np.ndarray:
        """
        Estimates the best fit parameters that define a warp field.
        :param warp: deformation field
        :return: model parameters
        """

        inv_b = np.linalg.pinv(self.basis)

        return np.hstack(
            (np.dot(inv_b, warp[1].flatten()), np.dot(inv_b, warp[0].flatten()))
        )

    def transform(self, p: np.ndarray) -> np.ndarray:
        """
        Applies an spline transformation to image coordinates.
        :param p: model parameters
        :return: deformation coordinates
        """

        px = np.array(p[0 : self.number_parameters])
        py = np.array(p[self.number_parameters : :])

        shape = self.coordinates.tensor[0].shape

        # FIXME: Inverse of a warp field needs to be derived and put in here,
        #        clearly a multiplication by -1 is not a good approach.

        return -1.0 * np.array(
            [
                np.dot(self.basis, py).reshape(shape),
                np.dot(self.basis, px).reshape(shape),
            ]
        )

    def jacobian(self, p: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the Jacobian matrix of the model.
        :param p: model parameters
        :return: derivative of the model with respect to the parameters
        """

        dx = np.zeros((self.coordinates.tensor[0].size, 2 * self.number_parameters))

        dy = np.zeros((self.coordinates.tensor[0].size, 2 * self.number_parameters))

        dx[:, 0 : self.number_parameters] = self.basis
        dy[:, self.number_parameters : :] = self.basis

        return dx, dy
