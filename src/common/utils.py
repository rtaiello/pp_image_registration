import os
import pickle
import shutil
from pathlib import Path
from typing import Optional

import dotenv
import numpy as np
import omegaconf
from matplotlib.image import imsave
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import wandb


def resize(image_pil, width, height):
    """
    Resize PIL image keeping ratio and using white background.

    :param image_pil: the image to resize
    :param width: the width of the new image
    :param height: the height of the new image
    :return: the resized image
    """
    ratio_w = width / image_pil.width
    ratio_h = height / image_pil.height
    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)
    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)
        resize_height = height
    image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert("L")


def get_name(cfg: DictConfig) -> str:
    return cfg.core.name


def get_group(cfg: DictConfig) -> str:
    group_name = f"{cfg.joint_computation.name}_{cfg.register.name}"
    return group_name


def save_resources(cfg: omegaconf.DictConfig, displacement: np.ndarray) -> None:
    """
    Save the resources used by the experiment.
    :param cfg:
    :param displacement:
    :return:
    """
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (Path(wandb.run.dir) / "hparams.yaml").write_text(yaml_conf)
    with open((Path(wandb.run.dir) / "displacement.npy"), "wb") as handle:
        np.save(handle, displacement)


def mse(displacement_clear, displacement_pp, pixel_size=3.18 * 2.02):

    displacement_diff = displacement_clear - displacement_pp
    norm = np.sqrt((displacement_diff[0]) ** 2 + (displacement_diff[1]) ** 2)
    avg = np.mean(norm) * pixel_size
    error = np.sqrt(avg)
    #error *= pixel_size
    return error

def log_statistics(
        logger,
        iterations,
        error,
        disp_error,
        megabytes_party_1,
        megabytes_party_2,
        times_party_1,
        times_party_2,
):
    """
    Log the statistics of the experiment.
    :param logger:
    :param iterations:
    :param error:
    :param megabytes_party_1:
    :param megabytes_party_2:
    :param times_party_1:
    :param times_party_2:
    :return:
    """

    logger.log(
        {
            "Time - Party 1 (s)": sum(times_party_1),
            "Time - Party 2 (s)": sum(times_party_2),
            "Comm - Party 1 (MB)": sum(megabytes_party_1),
            "Comm - Party 2 (MB)": sum(megabytes_party_2),
            "Intensity error (SSD)": error,
            "Displacement error (RMSE) mm": disp_error,
            "Num. Iterations": sum(iterations),
            "Time for one iteration - Party 1 (s)": sum(times_party_1) / sum(iterations),
            "Time for one iteration - Party 2 (s)": sum(times_party_2) / sum(iterations),
            "Comm for one iteration - Party 1 (MB)": sum(megabytes_party_1) / sum(iterations),
            "Comm for one iteration - Party 2 (MB)": sum(megabytes_party_2) / sum(iterations),
        }
    )


def get_env(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable

    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


STATS_KEY: str = "stats"

# Load environment variables
load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
DATA_ROOT: Path = Path(get_env("DATA_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)
