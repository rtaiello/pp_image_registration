import numpy as np
import tenseal as ts

# always compute on the second split set


class Party2v1:
    def __init__(self, template, n_split, n_threads):
        self.n = template.shape[0]
        self.template = template
        self.template_split = [t for t in np.hsplit(self.template.flatten(), n_split)]
        self.n_split = n_split
        self.n_threads = n_threads

        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=4096,
            coeff_mod_bit_sizes=[30, 24, 24, 30],
            n_threads=n_threads,
            encryption_type=ts.ENCRYPTION_TYPE.SYMMETRIC,
        )
        precision = 24
        self.context.global_scale = 2**precision
        self.context.generate_galois_keys()

    def get_serialized_context(self):
        _serialized_context = self.context.serialize(
            save_public_key=False,
            save_secret_key=False,
            save_galois_keys=False,
            save_relin_keys=False,
        )
        return _serialized_context

    def get_template_split_to_p1(self):
        """
        Split and encrypt J
        Returns:
            split & enc version of J
        """
        return [ts.ckks_vector(self.context, t.flatten()) for t in self.template_split]

    def dec_partial_res_from_p1(self, res_from_p1):
        """
        Decrypted the result got from P1
        Parameters
        ----------
        res_from_p1 : enc result from P1

        Returns
        -------
        Cleartext result to P1
        """
        res_from_p1.link_context(self.context)
        return np.array(res_from_p1.decrypt()).T
