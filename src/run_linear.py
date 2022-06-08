import multiprocessing
import os
from typing import List, Optional

import hydra
import numpy as np
import omegaconf
from matplotlib.image import imsave
from omegaconf import DictConfig

import wandb
from src.common.utils import (
    PROJECT_ROOT,
    get_group,
    get_name,
    log_statistics,
    save_resources,
    mse,
)
from src.models.affine import Affine
from src.registers.abstract_register import AbstractRegister, OptStep, RegisterData

multiprocessing.set_start_method("fork")


def run(cfg: DictConfig) -> None:
    template_path: str = cfg.data.template
    hydra.utils.log.info(f"src.run_linear.py - Loading template: {template_path}")

    image_path: str = cfg.data.moving
    hydra.utils.log.info(f"src.run_linear.py - Loading moving: {image_path}")
    print(cfg.data.preprocessing)
    image, template = hydra.utils.call(
        cfg.data.preprocessing, image_path, template_path
    )

    scale: float = cfg.data.scale

    template = template * scale
    image = image * scale

    hydra.utils.log.info(f"src.run_linear.py - Instantiating <{cfg.register._target_}>")
    logger = None
    if not cfg.logging.debug:
        hydra.utils.log.info(f"src.run_linear.py - Instantiating WANDB Logger")
        logger = wandb.init(
            **cfg.logging.wandb,
            project="affine",
            name=get_name(cfg),
            group=get_group(cfg),
        )
    p_hat: Optional[np.ndarray] = None
    down_sampled_image: Optional[np.ndarray] = None
    step: Optional[OptStep] = None

    iterations: List[int] = list()
    times_party_1: List[float] = list()
    times_party_2: List[float] = list()
    megabytes_party_1: List[float] = list()
    megabytes_party_2: List[float] = list()

    for item in cfg.data.iterations.items():
        factor = item[0]
        cfg.register.max_iter = (
            item[1]["max_iter"] if item[1]["max_iter"] != -1 else cfg.register.max_iter
        )
        cfg.register.max_bad = (
            item[1]["max_bad"] if item[1]["max_bad"] != -1 else cfg.register.max_bad
        )

        hydra.utils.log.info(
            f"src.run_linear.py  - Running <{cfg.model._target_}> - Factor {factor}"
        )

        if p_hat is not None:
            scale = down_sampled_image.coords.spacing / factor
            p_hat = Affine.scale(p_hat, scale)

        down_sampled_image: np.ndarray = template
        down_sampled_image: RegisterData = RegisterData(down_sampled_image).down_sample(
            factor
        )

        register: AbstractRegister = hydra.utils.instantiate(
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
        ) = register.execute(params=p_hat, verbose=cfg.logging.debug)
        iterations.append(len(searches))
        times_party_1.append(current_time_party_1)
        times_party_2.append(current_time_party_2)
        megabytes_party_1.append(current_megabytes_party_1)
        megabytes_party_2.append(current_megabytes_party_2)
        hydra.utils.log.info(
            f"src.run_linear.py  - Finished <{cfg.model._target_}> - Factor {factor}"
        )
        p_hat = step.params
    hydra.utils.log.info(
        f"src.run_linear.py - Writing warped image: {cfg.data.dir}/warped_affine.png"
    )
    os.makedirs(f"{cfg.data.dir}/{cfg.joint_computation.name}/{cfg.register.name}", exist_ok=True)
    np.save(f"{cfg.data.dir}/{cfg.joint_computation.name}/{cfg.register.name}/displacement.npy", step.displacement)
    imsave(
        f"{cfg.data.dir}/{cfg.joint_computation.name}/{cfg.register.name}/warped_affine.png",
        step.warped_image / scale,
        cmap="gray",
    )
    hydra.utils.log.info(
        f"src.run_linear.py  \n"
        f"Total Time - Party 1 (s): {sum(times_party_1)}  - Total Time - Party 2 (s): {sum(times_party_2)}  \n "
        f"Total Comm - Party 1 (MB): {sum(megabytes_party_1)}  - Total Comm - Party 2 (MB): {sum(megabytes_party_1)}"
    )
    disp_error = 0
    if cfg.joint_computation.name != "clear":
        y = np.load(f"{cfg.data.dir}/clear/{cfg.register.name}/displacement.npy")
        y_hat = step.displacement
        disp_error = mse(displacement_clear=y, displacement_pp=y_hat, pixel_size=3.18 * 2.02)
        hydra.utils.log.info(f"Displacement Error: {disp_error}")
    if not cfg.logging.debug:
        error = step.error
        log_statistics(
            logger,
            iterations,
            error,
            disp_error,
            megabytes_party_1,
            megabytes_party_2,
            times_party_1,
            times_party_2,

        )

        save_resources(cfg, step.displacement)

        logger.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    for i in range(0, 1):
        cfg.core.name = f"run_{i}"
        run(cfg)


if __name__ == "__main__":
    main()
