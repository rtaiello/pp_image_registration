""" A top level registration module """

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import hydra
import numpy as np
import scipy.ndimage as nd

from src.common.coordinates import Coordinates2D
from src.joint_computations.abstract_jc import AbstractJC
from src.metrics.abstract_metric import AbstractMetric
from src.models.abstract_model import AbstractModel
from src.samplers.abstract_sampler import AbstractSampler


class RegisterData:
    def __init__(
        self,
        data: np.ndarray,
        coords: Optional[Coordinates2D] = None,
        features: Optional[dict] = None,
        spacing: float = 1.0,
        *args: list,
    ) -> None:
        """
        Container for registration metric.

        :param data: the image registration image values
        :param coords: the grid coordinates
        :param features: a mapping of unique ids to registration features
        :param spacing:
        """
        self._data = data.astype(np.double)

        if not coords:
            self._coords = Coordinates2D(
                [0, data.shape[0], 0, data.shape[1]], spacing=spacing
            )

        else:
            self._coords = coords

        self.features = features

    @property
    def coords(self) -> Coordinates2D:
        return self._coords

    @property
    def data(self) -> np.ndarray:
        return self._data

    def down_sample(self, factor: float = 1.0):
        """
        Down samples the RegisterData by a user defined factor. The ndimage
        zoom function is used to interpolate the image, with a scale defined
        as 1/factor.

        Spacing is used to infer the scale difference between images - defining
        the size of a pixel in arbitrary units (atm).

        :param factor: the scaling factor which is applied to image metric and coordinates
        :return: the parameter update vector
        """

        resampled = nd.zoom(self._data, 1.0 / factor)

        return RegisterData(resampled, spacing=factor)

    @staticmethod
    def _smooth(image: np.ndarray, variance: float) -> np.ndarray:
        """
        Gaussian smoothing using the fast-fourier-transform (FFT)
        :param image: input image
        :param variance: variance of the Gaussian kernel
        :return: an image convolved with the Gaussian kernel
        """

        return np.real(np.fft.ifft2(nd.fourier_gaussian(np.fft.fft2(image), variance)))

    def smooth(self, variance) -> None:
        """
        Smooth feature metric in place.
        :param variance: variance of the Gaussian kernel
        :return: see RegisterData.smooth
        """

        self._data = RegisterData._smooth(self._data, variance)


"""
    A container class for optimization steps.

    Attributes
    ----------
    warped_image: nd-array
        Deformed image.
    warp: nd-array
        Estimated deformation field.
    grid: nd-array
        Grid coordinates in tensor form.
    error: float
        Normalised fitting error.
    p: nd-array
        Model parameters.
    delta_p: nd-array
        Model parameter update vector.
    decreasing: boolean.
        State of the error function at this point.
    """


@dataclass
class OptStep:
    """
    A container class for optimization steps.
    """

    warped_image: np.ndarray
    warp: AbstractModel
    grid: np.ndarray
    error: np.ndarray
    params: np.ndarray
    delta_p: np.ndarray
    decreasing: bool
    template: np.ndarray
    image: np.ndarray
    displacement: np.ndarray


class AbstractRegister(ABC):
    """
    A registration class for estimating the deformation model parameters that
    best solve:

    | :math:`f( W(I;p), J )`
    |
    | where:
    |    :math:`f`     : is a similarity metric.
    |    :math:`W(x;p)`: is a deformation model (defined by the parameter set p).
    |    :math:`I`     : is an input image (to be deformed).
    |    :math:`J`     : is a template (which is a deformed version of the input).

    Notes:
    ------

    Solved using a modified gradient descent algorithm.

    [0] Levernberg-Marquardt algorithm,
           http://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm

    Attributes
    ----------
    model: class
        A `deformation` model class definition.
    metric: class
        A `similarity` metric class definition.
    sampler: class
        A `sampler` class definition.
    """

    def __init__(
        self,
        image: np.ndarray,
        template: np.ndarray,
        model: AbstractModel,
        metric: AbstractMetric,
        sampler: AbstractSampler,
        joint_computation: AbstractJC,
        max_bad: int = 5,
        max_iter: int = 200,
        down_sample_factor: int = 1.0,
        logger=None,
        *args,
        **kwargs,
    ):
        """
         A registration class for estimating the deformation model parameters that
        best solve:

        | :math:`f( W(I;p), J )`
        |
        | where:
        |    :math:`f`     : is a similarity metric.
        |    :math:`W(x;p)`: is a deformation model (defined by the parameter set p).
        |    :math:`I`     : is an input image (to be deformed).
        |    :math:`J`     : is a template (which is a deformed version of the input).


        Solved using a modified gradient descent algorithm.

        [0] Levernberg-Marquardt algorithm,
               http://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm
        :param image: moving image
        :param template: template image
        :param model: a deformation model class
        :param metric: a similarity metric class
        :param sampler: a sampler (interpolator)
        :param joint_computation: a joint computation class
        :param max_bad: number of consecutive bad steps before stopping
        :param max_iter: maximum number of iterations
        :param down_sample_factor: down sampling factor
        :param logger: a logger
        """

        self.image: RegisterData = RegisterData(image).down_sample(down_sample_factor)
        self.template: RegisterData = RegisterData(template).down_sample(
            down_sample_factor
        )

        self.model = hydra.utils.instantiate(model, coordinates=self.image.coords)
        self.sampler = hydra.utils.instantiate(sampler, coordinates=self.image.coords)
        self.metric = hydra.utils.instantiate(metric)
        self.joint_computation = hydra.utils.instantiate(joint_computation)
        self.max_bad = max_bad
        self.max_iter = max_iter
        self.logger = logger

    @abstractmethod
    def _delta_params(
        self,
        jacobian: np.ndarray,
        grad_norm: np.ndarray,
        warped_image: np.ndarray,
        template_image: np.ndarray,
        error: float,
        alpha: float,
        params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes the parameter update.

        :param jacobian: the (d_error/d_params) the relationship between image differences and model parameters.
        :param grad_norm: the gradient norm
        :param warped_image: warped image
        :param template_image: template image
        :param error:
        :param alpha: dampening factor
        :param params: model parameters
        :return: the parameter update vector
        """

    pass

    @staticmethod
    def _dampening(alpha: float, decreasing: bool) -> float:
        """
        Computes the adjusted dampening factor.
        :param alpha: the current dampening factor
        :param decreasing: conditional on the decreasing error function
        :return: the adjusted dampening factor
        """
        return alpha / 10.0 if decreasing else alpha * 10.0

    def execute(
        self,
        params: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
        displacement: Optional[np.ndarray] = None,
        verbose: bool = False,
        decreasing: bool = True,
    ) -> Tuple[OptStep, List[OptStep], float, float, float, float]:
        """
        Executes the registration algorithm.

        :param params: initial model parameters
        :param alpha: initial dampening factor
        :param displacement: initial displacement
        :param verbose: verbose mode
        :param decreasing: conditional on the decreasing error function
        :return: the optimal step, the list of steps, the computation time of party 1 and 2, the bandwidth demanded
        by party 1 and 2
        """

        if self.image.coords.spacing != self.template.coords.spacing:
            raise ValueError("Coordinate systems differ.")

        # Initialize the models, metric and sampler.

        if displacement is not None:
            # Account for difference warp resolutions.
            scale = (
                (self.image.data.shape[0] * 1.0) / displacement.shape[1],
                (self.image.data.shape[1] * 1.0) / displacement.shape[2],
            )

            # Scale the displacement field and estimate the model parameters,
            # refer to test_CubicSpline_estimate
            scaled_displacement = (
                np.array(
                    [nd.zoom(displacement[0], scale), nd.zoom(displacement[1], scale)]
                )
                * scale[0]
            )

            # Estimate p, using the displacement field.
            params = self.model.estimate(-1.0 * scaled_displacement)

        params = self.model.identity if params is None else params
        delta_p = np.zeros_like(params)

        # Dampening factor.
        alpha = alpha if alpha is not None else 1e-4

        # Variables used to implement a back-tracking algorithm.
        searches: List[OptStep] = []
        num_bad_steps: int = 0
        best_step: Optional[OptStep] = None

        for iteration in range(0, self.max_iter):

            warp = self.model.warp(params)
            # Sample the image using the inverse warp.

            warped_image = RegisterData._smooth(
                self.sampler.f(self.image.data, warp).reshape(self.image.data.shape),
                0.5,
            )
            # warped_image = self.sampler.f(self.image.data, warp).reshape(self.image.data.shape)

            e = self.metric.error(warped_image, self.template.data)
            # Cache the optimization step.
            search_step: OptStep = OptStep(
                error=np.abs(e).sum() / np.prod(self.image.data.shape),
                params=params.copy(),
                delta_p=delta_p.copy(),
                grid=self.image.coords.tensor.copy(),
                warp=warp.copy(),
                displacement=self.model.transform(params),
                warped_image=warped_image.copy(),
                template=self.template.data,
                image=self.image.data,
                decreasing=decreasing,
            )

            # Update the current best step.
            best_step: OptStep = search_step if best_step is None else best_step

            if verbose:
                print(
                    f"\n iteration  : {iteration} \n parameters : {[param for param in search_step.params]} \n error      : {search_step.error} \n"
                )
            if self.logger:
                self.logger.log({"error": search_step.error})
            # Append the search step to the search.
            searches.append(search_step)

            if len(searches) > 1:
                # Check if the search step is improving.
                search_step.decreasing = search_step.error < best_step.error

                alpha = self._dampening(alpha, search_step.decreasing)

                if search_step.decreasing:

                    best_step = search_step
                else:
                    num_bad_steps += 1

                    if num_bad_steps > self.max_bad:
                        if verbose:
                            hydra.utils.log.info(
                                f"src.registers.abstract_register.py\n"
                                "Optimization break, maximum number "
                                "of bad iterations exceeded."
                            )
                        break
                    params = best_step.params.copy()

            # Computes the derivative of the error with respect to model
            # parameters.

            jacobian, grad_norm = self.metric.get_jacobian(
                self.model, warped_image, params
            )

            # Compute the parameter update vector.
            delta_p = self._delta_params(
                jacobian=jacobian,
                grad_norm=grad_norm,
                warped_image=warped_image,
                template_image=self.template.data,
                error=e,
                alpha=alpha,
                params=params,
            )

            # Evaluate stopping condition:
            if np.dot(delta_p.T, delta_p) < 1e-4:
                hydra.utils.log.info(
                    "f src.register_abstract_register.py \n" "Stopping condition"
                )
                break

            # Update the estimated parameters.
            params += delta_p

        return (
            best_step,
            searches,
            self.joint_computation.party_1_total_time,
            self.joint_computation.party_2_total_time,
            self.joint_computation.party_1_total_megabytes,
            self.joint_computation.party_2_total_megabytes,
        )

    @staticmethod
    def next_power_of_2(n):
        """
        Returns the next power of 2 greater than or equal to n.
        :param n: the number to be rounded up
        :return: the next power of 2
        """

        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n += 1
        return n
