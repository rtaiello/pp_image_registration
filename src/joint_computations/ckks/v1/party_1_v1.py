from multiprocessing import Manager, Process
from typing import List, Optional, Union

import numpy as np
import tenseal as ts

# always compute on the first split set


class Party1v1:
    def __init__(self, s, n_split):
        self.n = s.shape[0]
        self.m = s.shape[1]
        self.s: np.ndarray = s
        self.context = None
        self.s_split: Union[List[np.ndarray], List[ts.CKKSVector]] = [
            t for t in np.hsplit(self.s, n_split)
        ]
        self.n_split = n_split

    def compute_res_enc_split(
        self, template_split_from_p2, n_threads, serialized_context_from_p2
    ):
        if n_threads == 1:
            return self._compute_res_enc_split_parallel(
                template_split_from_p2, serialized_context_from_p2
            )
        return self._compute_res_enc_split_single(template_split_from_p2)

    def _compute_res_enc_split_single(
        self, template_split_from_p2: List[ts.CKKSVector]
    ):
        """
        Compute the enc result of the multiplication
        :param template_split_from_p2: list of enc split vectors template  from p2
        :param n_split:
        :return: result of the multiplication
        """

        return sum(
            [template_split_from_p2[i] @ self.s_split[i].T for i in range(self.n_split)]
        )

    def _compute_res_enc_split_parallel(
        self, template_split_from_p2, serialized_context_from_p2
    ):
        """
        Function used to parallelize the matrix multiplication
        Parameters
        ----------
        template_from_p2 : ts tensors list from P2

        Returns
        -------
        Partial encrypted result
        """
        context_from_p2 = ts.context_from(serialized_context_from_p2)

        proc_array = []
        shared_list = Manager().list()
        for i in range(self.n_split):
            p = Process(
                target=self.__ex_matrix_mult,
                args=(template_split_from_p2[i], self.s_split[i].T, shared_list),
            )
            proc_array.append(p)
        for p in proc_array:
            p.start()

        for p in proc_array:
            p.join()

        part_res = list()
        # Deserialization
        for ser_ts in shared_list:
            part_res.append(ts.ckks_vector_from(context=context_from_p2, data=ser_ts))

        return sum(part_res)

    def __ex_matrix_mult(self, array_enc, mat, shared_res):
        """
        target method for parallelization
        Parameters
        ----------
        array_enc : ts tensor
        mat : cleartext matrix
        shared_res : shared list for result

        Returns
        -------

        """

        res = array_enc @ mat
        # The ts tensor needs to be serialized to be saved and passed between functions
        shared_res.append(res.serialize())
