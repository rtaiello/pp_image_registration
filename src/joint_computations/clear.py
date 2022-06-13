from time import time

import numpy as np

from src.joint_computations.abstract_jc import AbstractJC


class Clear(AbstractJC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def joint_mm_ssd(self, s: np.ndarray, template: np.ndarray) -> np.ndarray:
        start: float = time()
        result: np.ndarray = np.matmul(s, template)
        end: float = time()
        self._party_1_total_time += end - start
        self._party_2_total_time = self._party_1_total_time
        return result

    def __str__(self) -> str:
        return "clear"
