from abc import ABC, abstractmethod

import numpy as np


class AbstractJC(ABC):
    def __init__(self, *args, **kwargs) -> None:
        self._party_1_total_megabytes: float = 0
        self._party_2_total_megabytes: float = 0
        self._party_1_total_time: float = 0
        self._party_2_total_time: float = 0

    @abstractmethod
    def joint_mm_ssd(self, s: np.ndarray, template: np.ndarray) -> np.ndarray:
        pass

    @property
    def party_1_total_megabytes(self) -> float:
        return self._party_1_total_megabytes / 2**20

    @property
    def party_2_total_megabytes(self) -> float:
        return self._party_2_total_megabytes / 2**20

    @property
    def party_1_total_time(self) -> float:
        return self._party_1_total_time

    @property
    def party_2_total_time(self) -> float:
        return self._party_2_total_time
