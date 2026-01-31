import argparse
import importlib
import numpy as np
import random, torch
from functools import reduce
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, Qwen3Config

from quantize import QuantConfig

import json
import os
model2path = json.load(open(os.path.join(os.path.dirname(__file__), "model2path.json"), "r"))


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument('--model_name', type=str, help="model to load")
    parser.add_argument('--use_fp16', action="store_true", default=False, help="Whether to use the original FP16 model.")
    return parser


def add_quant_args(parser):
    parser.add_argument('--w_bits', type=int, default=16, help="Number of bits for weight quantization.")
    parser.add_argument('--w_dtype', type=str, default="fp16", help="Weight data type for quantization.")
    parser.add_argument('--w_outlier', type=float, default=8.0, help="Outlier special value for Razer weight")
    parser.add_argument('--a_bits', type=int, default=16, help="Number of bits for activation quantization.")
    parser.add_argument('--a_dtype', type=str, default="fp16", help="Activation data type for quantization.")
    parser.add_argument('--k_bits', type=int, default=16, help="Number of bits for key quantization.")
    parser.add_argument('--v_bits', type=int, default=16, help="Number of bits for value quantization.")
    parser.add_argument('--w_groupsize', type=int, default=-1, help="Group size for weight quantization.")
    parser.add_argument('--a_groupsize', type=int, default=-1, help="Group size for activation quantization.")
    parser.add_argument('--k_groupsize', type=int, default=-1, help="Group size for key quantization.")
    parser.add_argument('--v_groupsize', type=int, default=-1, help="Group size for value quantization.")
    
    ############### KV-cache Quantization Arguments ###############
    parser.add_argument("--kv_quant", action="store_true", default=False, help="Whether to quantize KV-cache.")

    return parser
    

def get_quant_config(args):
    quant_config = QuantConfig(
        w_bits=args.w_bits,
        w_dtype=args.w_dtype,
        w_outlier=args.w_outlier,
        a_bits=args.a_bits,
        a_dtype=args.a_dtype,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        w_groupsize=args.w_groupsize,
        a_groupsize=args.a_groupsize,
        k_groupsize=args.k_groupsize,
        v_groupsize=args.v_groupsize,
        ############### KV-cache Quantization Arguments ###############
        kv_quant=args.kv_quant
    )
    return quant_config


# Set seed for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name, quant_config=None, device_map="auto", use_fp16: bool=False):
    """
    Args:
        model_name: The model to be evaluated.
        quant_config: The quantization configuration. Will be discarded if "use_fp16=True".
        device_map: "cpu" or "cuda" or "auto".
        use_fp16: If set to True, then evaluate the original FP16 model.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    model_path_fp16 = model2path[model_name]

    if 'llama' in model_path_fp16.lower():
        config = LlamaConfig.from_pretrained(model_path_fp16)

        if use_fp16:
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(
                model_path_fp16,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=device_map
            )
        else: 
            from models.qmodule_llama import QuantLlamaForCausalLM
            model = QuantLlamaForCausalLM.from_pretrained(
                model_path_fp16,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                quant_config=quant_config
            )

    elif 'qwen3' in model_path_fp16.lower():
        config = Qwen3Config.from_pretrained(model_path_fp16)

        if use_fp16:
            from transformers import Qwen3ForCausalLM
            model = Qwen3ForCausalLM.from_pretrained(
                model_path_fp16,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=device_map
            )
        else:         
            from models.qmodule_qwen3 import QuantQwen3ForCausalLM
            model = QuantQwen3ForCausalLM.from_pretrained(
                model_path_fp16,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                quant_config=quant_config
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path_fp16,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path_fp16,
        trust_remote_code=True,
    )

    model.eval() 

    return model, tokenizer
