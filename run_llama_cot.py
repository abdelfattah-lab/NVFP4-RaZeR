# Import necessary modules
import torch

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from lm_eval.tasks import TaskManager

import argparse
from tqdm import tqdm
from loguru import logger
import os
import json

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

from utils import (
    load_model_and_tokenizer, 
    add_common_args, 
    add_quant_args, 
    get_quant_config,
    set_seed,
    model2path
)
from quantize import quant_weight


def run_lm_eval(
    model, tokenizer, args, max_length=4096
):
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    task_manager = TaskManager()
    results = {}

    # Setting task_manager to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in lm_eval/tasks.
    # simple_evaluate will instantiate its own task_manager is the it is set to None here.
    logger.info(f"Evaluation Task(s): {args.tasks}")

    for task_name in args.tasks:
        print(task_name)
        task_results = lm_eval.simple_evaluate( # call simple_evaluate
            model=lm_obj,
            tasks=task_name,
            batch_size=args.batch_size,
            task_manager=task_manager,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
        )["results"]

        results.update(task_results)
        print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))

    return results


if __name__ == '__main__':
    # Ignore all warnings
    warnings.filterwarnings("ignore")
    # Set random seed
    set_seed(42)

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_quant_args(parser)
    parser.add_argument("--tasks", type=lambda s: [item for item in s.split(',')], default=[], help="Task to be evaluated")
    parser.add_argument("--output_dir", type=str, default="results/acc", help="Output Directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
    args = parser.parse_args()  
    
    quant_config = get_quant_config(args)
    model_name = args.model_name
    model_name_or_path = model2path[model_name]

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")
    logger.info(f"#################### Model Info ####################")
    logger.info(f"* Model: {model_name_or_path}")
    logger.info(f"* Tasks: {args.tasks}")

    logger.info("#################### Creating output directory ... ####################")
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.use_fp16:
        output_file_name = "Baseline_FP16.txt"
    else:
        output_file_name = f"w{args.w_bits}_g{args.w_groupsize}_{args.w_dtype}__a{args.a_bits}_g{args.a_groupsize}_{args.a_dtype}.json"
    output_file_path = os.path.join(output_dir, f"{output_file_name}")
    # check if result file exists
    if os.path.isfile(output_file_path):
        print(f'Found existing output file  {output_file_name}  for this experiment. Exit!\n\n')
        exit()
    print(f'Results will be saved to the output file:  {output_file_name}\n')

    logger.info(f"#################### Quantization Info ####################")
    print(f"==================================================")
    print(f"Weight Quantization Data Type:      {quant_config.w_dtype}")
    print(f"Weight Quantization Bits:           {quant_config.w_bits}")
    print(f"Weight Quantization Group Size:     {quant_config.w_groupsize}")
    print()
    print(f"Activation Quantization Data Type:  {quant_config.a_dtype}")
    print(f"Activation Quantization Bits:       {quant_config.a_bits}")
    print(f"Activation Quantization Group Size: {quant_config.a_groupsize}")
    print(f"==================================================")

    logger.info("#################### Loading model and tokenizer ... ####################")
    model, tokenizer = load_model_and_tokenizer(model_name, quant_config=quant_config, use_fp16=args.use_fp16)
    quant_weight(model, quant_config)
    
    logger.info("#################### Start running LM_Eval zero-shot evaluation ... #################### ")
    results = run_lm_eval(model, tokenizer, args)
    
    # Save results to JSON file
    with open(output_file_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file_path} \n\n")
    