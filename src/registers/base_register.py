from typing import Optional

import numpy as np

from src.registers.abstract_register import AbstractRegister


class BaseRegister(AbstractRegister):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    def _delta_params(
        self,
        jacobian: np.ndarray,
        grad_norm: np.ndarray,
        warped_image: np.ndarray,
        template_image: np.ndarray,
        error: np.ndarray,
        alpha: float,
        params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the update of the parameters.
        :param jacobian:
        :param grad_norm:
        :param warped_image:
        :param template_image:
        :param error:
        :param alpha:
        :param params:
        :return:
        """

        template_image = template_image.reshape(-1, 1)
        warped_image = warped_image.reshape(-1, 1)
        h: np.ndarray = np.dot(jacobian.T, jacobian)

        h += np.diag(alpha * np.diagonal(h))

        h_inv: np.ndarray = np.linalg.inv(h)

        private_computation: np.ndarray = jacobian.T @ warped_image
        joint_computation: np.ndarray = self.joint_computation.joint_mm_ssd(
            s=jacobian.T, template=template_image
        )
        result: np.ndarray = (
            h_inv @ (private_computation - joint_computation)
        ).flatten()
        return result
