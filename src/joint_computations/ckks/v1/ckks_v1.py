from time import time
from typing import Optional, List

import hydra
import numpy as np
import tenseal as ts

from src.joint_computations.abstract_jc import AbstractJC
from src.joint_computations.ckks.v1.party_1_v1 import Party1v1
from src.joint_computations.ckks.v1.party_2_v1 import Party2v1


class CKKSv1(AbstractJC):
    def __init__(self, dim_split_vectors: int, n_threads: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim_split_vectors = dim_split_vectors
        self.party_1: Optional[Party1v1] = None
        self.party_2: Optional[Party2v1] = None
        self.n_split: Optional[int] = None
        self.n_threads: int = n_threads
        self.first_iteration: bool = True

    def one_side_computation_protocol(self):
        """
        One side computation P1
        """
        hydra.utils.log.info(
            f"src.joint_computations.ckks.ckks_v1.py - Num Split:{self.n_split}"
        )
        start = time()
        template_split_from_p2_enc = self.party_2.get_template_split_to_p1()
        end = time()

        if self.first_iteration:
            self.first_iteration = False
            # Party 2 sends the template split to Party 1 on the first iteration

            template_enc_bytes: int = sum(
                [len(t.serialize()) for t in template_split_from_p2_enc]
            )

            self._party_2_total_megabytes += template_enc_bytes
            self._party_2_total_time += end - start
            hydra.utils.log.info(f"src.joint_computations.ckks.ckks_v1.py - "
                                 f"Party2 Enc Time: {end - start} (s) - Comm. : {template_enc_bytes/2**20} (MB)")
        serialized_context_from_p2 = self.party_2.get_serialized_context()

        start = time()
        result_one_side = self.party_1.compute_res_enc_split(
            template_split_from_p2=template_split_from_p2_enc,
            n_threads=self.n_threads,
            serialized_context_from_p2=serialized_context_from_p2,
        )
        end = time()
        self._party_1_total_time += end - start
        self._party_1_total_megabytes += len(result_one_side.serialize())
        hydra.utils.log.info(f"src.joint_computations.ckks.v1.ckks_v1.py - "
                             f"Party 1 sent encrypted result to Party 2 - "
                             f"Time: {end - start} (s) - Comm.: {len(result_one_side.serialize())/2**20} (MB)")
        start = time()
        dec_result_one_side = self.party_2.dec_partial_res_from_p1(result_one_side)
        end = time()
        self._party_2_total_time += end - start
        self._party_2_total_megabytes += dec_result_one_side.nbytes
        hydra.utils.log.info(f"src.joint_computations.ckks.v1.ckks_v1.py - Party 2 decrypted result from Party 1 - "
                             f"Time: {end - start} (s) - Comm.: {dec_result_one_side.nbytes/2**20} (MB)")

        return dec_result_one_side.reshape(-1, 1)

    def joint_mm_ssd(self, s: np.ndarray, template: np.ndarray) -> np.ndarray:
        self.n_split = len(template) // self.dim_split_vectors
        self.party_1 = Party1v1(s=s, n_split=self.n_split)
        self.party_2 = Party2v1(
            template=template, n_split=self.n_split, n_threads=self.n_threads
        )

        return self.one_side_computation_protocol()
