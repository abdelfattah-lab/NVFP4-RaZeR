from typing import Optional

class QuantConfig(dict):
    def __init__(
        self,
        # general quantization parameters
        w_bits: int=16,
        w_dtype: str="fp16",
        w_outlier: float=8.0,
        a_bits: int=16,
        a_dtype: str="fp16",
        k_bits: int=16,
        v_bits: int=16,
        w_groupsize: int=-1,
        a_groupsize: int=-1,
        k_groupsize: int=-1,
        v_groupsize: int=-1,
        kv_quant: bool=False,  # If True, then quantize KV-cache
    ):
        for nbits in [w_bits, k_bits, v_bits]:
            assert (nbits is None) or (nbits in [3, 4, 5, 6, 8, 16]), \
                f'Invalid precision \"{nbits}\" provided for weight / KV-cache. Allowed precisions are {{3, 4, 6, 8, 16}}'
        for nbits in [a_bits]:
            assert (nbits is None) or (nbits in [4, 8, 16]), \
                f'Invalid precision \"{nbits}\" provided for activation / query. Allowed precisions are {{8, 16}}'

        self.w_bits = w_bits
        self.w_dtype = w_dtype
        self.w_outlier = w_outlier
        self.a_bits = a_bits
        self.a_dtype = a_dtype
        self.k_bits = k_bits
        self.v_bits = v_bits

        self.w_groupsize = w_groupsize
        self.a_groupsize = a_groupsize
        self.k_groupsize = k_groupsize
        self.v_groupsize = v_groupsize

        self.kv_quant = kv_quant
    
    def __repr__(self):
        return repr(self.__dict__)
