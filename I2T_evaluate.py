import torch
import json
import argparse
from evals import eval
import transformers


metrics = {
    "textocr": "Acc",
    "operator_induction": "Acc",
    "open_mi": "Acc",
    "clevr": "Acc",
    'operator_induction_interleaved': "Acc",
    'matching_mi': "Acc",
}


def parse_args():
    parser = argparse.ArgumentParser(description='I2T ICL Evaluation')

    parser.add_argument('--dataDir', default='./VL-ICL', type=str, help='Data directory.')
    parser.add_argument('--dataset', default='operator_induction', type=str, choices=['operator_induction', 'textocr', 'open_mi', 
                                                                             'clevr','operator_induction_interleaved', 'matching_mi',])
    parser.add_argument("--engine", "-e", choices=["openflamingo", "otter-llama", "llava16-7b", "qwen-vl", "qwen-vl-chat", "qwen2.5-vl-3b", "qwen2.5-vl-7b", 'internlm-x2', 
                                                   'emu2-chat', 'idefics-9b-instruct', 'idefics-80b-instruct', 'gpt4v', 'llava-onevision-7b',
                                                   'llava-onevision-0.5b', 'phi-3.5-vision'],
                        default=["llava16-7b"], nargs="+")
    parser.add_argument('--max-new-tokens', default=15, type=int, help='Max new tokens for generation.')
    parser.add_argument('--n_shot', default=[0, 1, 2, 4, 8], nargs="+", help='Number of support images.')
    parser.add_argument('--wo_img', action='store_true', help='Whether to evaluate without image.')
    parser.add_argument('--wo_query_img', action='store_true', help='Whether to evaluate without query image.')
    parser.add_argument('--contain', action='store_true', help='Whether to evaluate with contain.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.wo_img:
        result_files = [f"results/wo_img/{args.dataset}/{engine}_{shot}-shot.json" for engine in args.engine for shot in args.n_shot]
    elif args.wo_query_img:
        result_files = [f"results/wo_query_img/{args.dataset}/{engine}_{shot}-shot.json" for engine in args.engine for shot in args.n_shot]
    else:
        result_files = [f"results/w_img/{args.dataset}/{engine}_{shot}-shot.json" for engine in args.engine for shot in args.n_shot]

    for result_file in result_files:
        engine, shot = result_file.split("/")[-1].replace(".json", "").split("_")
        with open(result_file, "r") as f:
            results_dict = json.load(f)
        if args.contain:
            score = eval.eval_scores_contain(results_dict, args.dataset)
        else:
            score = eval.eval_scores(results_dict, args.dataset)
        print(f'{args.dataset} {metrics[args.dataset]} of {engine} {shot}: ', f"{score * 100.0:.2f}", flush=True)