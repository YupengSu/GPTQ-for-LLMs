import time
import os
import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_llm(model_name, cache_dir="llm_weights", seqlen=2048):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto",
        trust_remote_code=True
    )
    model.seqlen = seqlen 
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--method', type=str, default='gptq',
        help='Method to use for quantization.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save_model', type=str, default='',
        help='Save the fake quantized model under this name.'
    )
    parser.add_argument(
        '--save_checkpoint', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        '--save_ppl', type=str, default='',
        help='Save perplexity results under this directory.'
    )
    parser.add_argument(
        '--save_zeroshot', type=str, default='',
        help='Save zero-shot results under this directory.'
    )
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Set cache directory for Huggingface datasets
    os.environ['HF_DATASETS_CACHE'] = os.path.join(os.getcwd(), 'datasets_cache')

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {model_name}")
    
    model = get_llm(args.model)
    model.eval()
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    device = torch.device("cuda:0")
    
    if 'llama' in model_name:
        from lib.llama import quant_gptq, quant_minmax, llm_pack3
    elif 'cogvlm' in model_name:
        from lib.cogvlm import quant_gptq, quant_minmax, llm_pack3
    elif 'internvl' in model_name:
        from lib.internvl import quant_gptq, quant_minmax, llm_pack3
    elif 'glm' in model_name:
        from lib.chatglm import quant_gptq, quant_minmax, llm_pack3

    if args.wbits < 16:
        if args.method == 'gptq':
            quantizers = quant_gptq(args, model, tokenizer, device)
        elif args.method == 'minmax':
            quantizers = quant_minmax(args, model, tokenizer, device)
    
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    if args.save_ppl:
        from lib.eval import eval_ppl
        ppl_test = eval_ppl(args, model, tokenizer, device)

        if not os.path.exists(args.save_ppl):
            os.makedirs(args.save_ppl)
        save_filepath = os.path.join(args.save_ppl, f"log_{args.wbits}.txt")
        with open(save_filepath, "w") as f:
            print(f"{'method':<15}{'actual_wbits':<15}{'wikitest2':<15}{'ptb':<15}{'c4':<15}", file=f, flush=True)
            print(f"{'gptq':<15}{args.wbits:<15.4f}{ppl_test[0]:<15.4f}{ppl_test[1]:<15.4f}{ppl_test[2]:<15.4f}", file=f, flush=True)
    
    if args.save_zeroshot:
        from lib.eval import eval_zero_shot
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, args.save_model, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

        df = pd.DataFrame(results).T
        df_str = df.to_string()

        if not os.path.exists(args.save_zeroshot):
            os.makedirs(args.save_zeroshot)
        save_filepath = os.path.join(args.save_zeroshot, f"log_{args.wbits}.txt")
        with open(save_filepath, "w") as f:
            print(df_str, file=f, flush=True)

    if args.save_checkpoint:
        model = model.to(torch.device("cpu"))
        llm_pack3(model, quantizers)
        os.makedirs(os.path.dirname(args.save_checkpoint), exist_ok=True)
        torch.save(model.state_dict(), args.save_checkpoint)
