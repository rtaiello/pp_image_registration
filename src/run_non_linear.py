import multiprocessing
from typing import List, Optional

import hydra
import numpy as np
import omegaconf
import scipy.ndimage as nd
from matplotlib.image import imsave
from omegaconf import DictConfig

import wandb
from src.common.utils import (
    PROJECT_ROOT,
    get_group,
    get_name,
    log_statistics,
    save_resources,
)
from src.registers.abstract_register import AbstractRegister, OptStep

multiprocessing.set_start_method("fork")


def run(cfg: DictConfig) -> None:
    template_path: str = cfg.data.template
    hydra.utils.log.info(f"src.run_non_linear.py - Loading template: {template_path}")

    image_path: str = cfg.data.warped_affine
    hydra.utils.log.info(f"src.run_non_linear.py - Loading image: {image_path}")

    image, template = hydra.utils.call(
        cfg.data.preprocessing, image_path, template_path
    )

    scale: float = cfg.data.scale

    template: np.ndarray = template * scale
    image: np.ndarray  = image * scale
    logger = None
    step: Optional[OptStep] = None

    if not cfg.logging.debug:
        hydra.utils.log.info(f"Instantiating WANDB Logger")
        logger = wandb.init(
            **cfg.logging.wandb,
            project="non_linear",
            name=get_name(cfg),
            group=get_group(cfg),
        )

    iterations: List[int] = list()
    times_party_1: List[float] = list()
    times_party_2: List[float] = list()
    megabytes_party_1: List[float] = list()
    megabytes_party_2: List[float] = list()

    displacement: Optional[np.ndarray] = None

    for item in cfg.data.iterations.items():
        factor: float = item[0]
        cfg.register.max_iter = (
            item[1]["max_iter"] if item[1]["max_iter"] != -1 else cfg.register.max_iter
        )
        cfg.register.max_bad = (
            item[1]["max_bad"] if item[1]["max_bad"] != -1 else cfg.register.max_bad
        )

        hydra.utils.log.info(
            f"src.run_linear.py  - Running <{cfg.model._target_}> - Factor {factor}"
        )

        register_cubic_spline: AbstractRegister = hydra.utils.instantiate(
            cfg.register,
            image=image,
            template=template,
            model=cfg.model,
            metric=cfg.metric,
            sampler=cfg.sampler,
            joint_computation=cfg.joint_computation,
            logger=logger,
            down_sample_factor=factor,
            _recursive_=False,
        )
        (
            step,
            searches,
            current_time_party_1,
            current_time_party_2,
            current_megabytes_party_1,
            current_megabytes_party_2,
        ) = register_cubic_spline.execute(
            displacement=displacement, verbose=cfg.logging.debug, decreasing=True
        )
        iterations.append(len(searches))

        times_party_1.append(current_time_party_1)
        times_party_2.append(current_time_party_2)
        megabytes_party_1.append(current_megabytes_party_1)
        megabytes_party_2.append(current_megabytes_party_2)

        displacement: np.ndarray = step.displacement

        hydra.utils.log.info(
            f"src.run_non_linear.py -  Finished <{cfg.model._target_}> Factor {factor}"
        )
    hydra.utils.log.info(
        f"src.run_non_linear.py - Writing warped image: {cfg.data.dir}/warped_affine.png"
    )

    imsave(
        f"{cfg.data.dir}/warped_cubic_spline.png",
        nd.rotate((step.warped_image / scale), angle=90, reshape=False),
        cmap="gray",
    )
    if not cfg.logging.debug:
        log_statistics(
            logger=logger,
            iterations=iterations,
            error=step.error,
            disp_error=displacement,
            megabytes_party_1=megabytes_party_1,
            megabytes_party_2=megabytes_party_2,
            times_party_1=times_party_1,
            times_party_2=times_party_2,
        )

        save_resources(cfg=cfg, displacement=step.displacement)

        wandb.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    for i in range(0, 1):
        cfg.core.name = f"run_{i}"
        run(cfg)


if __name__ == "__main__":
    main()
