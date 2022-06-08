""" A top level registration module """

import hydra
import numpy as np
import numpy.random

from src.joint_computations.abstract_jc import AbstractJC
from src.metrics.abstract_metric import AbstractMetric
from src.models.abstract_model import AbstractModel
from src.registers.abstract_register import AbstractRegister
from src.samplers.abstract_sampler import AbstractSampler


class GMSRegister(AbstractRegister):
    def __init__(self, percent_pixels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percent_pixels = percent_pixels
        self.num_iteration = 0

    def _delta_params(
        self,
        jacobian: np.ndarray,
        grad_norm: np.ndarray,
        warped_image: np.ndarray,
        template_image: np.ndarray,
        error,
        alpha,
        params=None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed=self.num_iteration)

        self.num_iteration += 1

        grad_norm_probabilities = grad_norm / sum(grad_norm)
        num_pixels = AbstractRegister.next_power_of_2(
            ((len(template_image) // self.percent_pixels) * 100)
        )
        hydra.utils.log.info(
            f"src.registers.gms_register.py - percent of used pixels: {self.percent_pixels}\n"
            f"- Original number of pixel: {np.prod(template_image.shape)}\n"
            f"- Number of pixels after GMS sampling: {num_pixels}"
        )
        random_idx = rng.choice(
            jacobian.shape[0], size=num_pixels, p=grad_norm_probabilities, replace=False
        )

        jacobian = jacobian[random_idx]
        template_image = template_image.reshape(-1, 1)[random_idx]
        warped_image = warped_image.reshape(-1, 1)[random_idx]

        h = np.dot(jacobian.T, jacobian)

        h += np.diag(alpha * np.diagonal(h))
        joint_computation = self.joint_computation.joint_mm_ssd(
            s=jacobian.T, template=template_image
        )

        h_inv = np.linalg.inv(h)

        private_computation = jacobian.T @ warped_image
        result = (h_inv @ (private_computation - joint_computation)).flatten()

        return result
