from time import time
from typing import List

import hydra
import numpy as np
import syft as sy
import torch

from src.joint_computations.abstract_jc import AbstractJC

PARTY_1 = 0
PARTY_2 = 1


class SPDZ(AbstractJC):
    def __init__(self, n_parties: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hook = sy.TorchHook(torch)
        self.parties: List[sy.VirtualWorker] = [
            sy.VirtualWorker(hook=hook, id=f"party_{i}") for i in range(n_parties)
        ]

        self.crypto_provider = sy.VirtualWorker(hook=hook, id="crypto_provider")
        self.parties[PARTY_1].log_msgs = True
        self.parties[1].log_msgs = True
        self.crypto_provider.log_msgs = True

    def joint_mm_ssd(self, s: np.ndarray, template: np.ndarray) -> np.ndarray:
        self.parties[PARTY_1].msg_history = list()
        self.parties[PARTY_2].msg_history = list()
        self.crypto_provider.msg_history = list()

        jacobian_t_torch = torch.tensor(s)
        template_torch = torch.tensor(template)
        start = time()
        jacobian_t_ptr = jacobian_t_torch.fix_prec(precision_fractional=3).share(
            self.parties[PARTY_1],
            self.parties[PARTY_2],
            crypto_provider=self.crypto_provider,
        )
        template_torch_ptr = template_torch.fix_prec(precision_fractional=3).share(
            self.parties[PARTY_1],
            self.parties[PARTY_2],
            crypto_provider=self.crypto_provider,
        )

        result_enc_ptr = jacobian_t_ptr @ template_torch_ptr
        result_dec: torch.Tensor = result_enc_ptr.get().float_precision()
        end = time()
        party_1_bytes = SPDZ._count_bytes(self.parties[PARTY_1])
        party_2_bytes = SPDZ._count_bytes(self.parties[PARTY_2])
        crypto_total_bytes = SPDZ._count_bytes(self.crypto_provider)

        self._party_1_total_megabytes += party_1_bytes + crypto_total_bytes
        self._party_2_total_megabytes += party_2_bytes + crypto_total_bytes
        self._party_1_total_time += end - start
        self._party_2_total_time = self._party_1_total_time

        hydra.utils.log.info(
            "src.joint_computations.spdz.SPDZ\n"
            f"Time (s) {(end - start)} Comm Party 1 (MB): {(party_1_bytes + crypto_total_bytes) / 2 ** 20} - Comm Party 2 "
            f"(MB): {(party_2_bytes + crypto_total_bytes) / 2 ** 20} ",
        )

        return result_dec.numpy()

    @staticmethod
    def _count_bytes(worker):
        """
        Counts the number of bytes. As messages in PySyft seem to be bytes objects we can use the length to determine
        the number of bytes per message:
        https://en.wikiversity.org/wiki/Python_Concepts/Bytes_objects_and_Bytearrays#bytes_objects
        :param worker: The worker.
        :return: The total bytes for this worker.
        """
        total_bytes = 0
        for msg in worker.msg_history:
            total_bytes += len(msg)
        return total_bytes
