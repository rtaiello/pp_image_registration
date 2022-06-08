from typing import List, Optional

import numpy as np


class Coordinates2D:
    def __init__(self, domain: List[int], spacing: Optional[float] = None):

        self.spacing: float = 1.0 if not spacing else spacing
        self.domain = domain
        self._tensor: np.ndarray = np.mgrid[0.0 : domain[1], 0.0 : domain[3]]

        self._homogenous: np.ndarray = np.zeros((3, self.tensor[0].size))
        self.homogenous[0]: np.ndarray = self.tensor[1].flatten()
        self.homogenous[1]: np.ndarray = self.tensor[0].flatten()
        self.homogenous[2]: float = 1.0

    @property
    def homogenous(self):
        return self._homogenous

    @property
    def tensor(self):
        return self._tensor
