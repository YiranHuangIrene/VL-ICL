import torch
import os
import json
import argparse
import gc
from utils import model_inference, utils, ICL_utils, load_models
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='I2T ICL Inference')

    parser.add_argument('--dataDir', default='./VL-ICL', type=str, help='Data directory.')
    parser.add_argument('--dataset', default='operator_induction', type=str, choices=['operator_induction', 'textocr', 'open_mi', 
                                                                             'clevr','operator_induction_interleaved', 'matching_mi',])
    parser.add_argument("--engine", "-e", choices=["openflamingo", "otter-llama", "llava16-7b", "llava16-13b", "llava16-13b-icl", "qwen-vl", "qwen-vl-chat", "qwen2.5-vl-3b", "qwen2.5-vl-7b", 'internlm-x2', 
                                                   'emu2-chat', 'idefics-9b-instruct', 'idefics-80b-instruct', 'gpt4v', 'llava-onevision-7b',
                                                   'llava-onevision-0.5b','phi-3.5-vision','phi-3-vision'],
                        default=["llava16-7b"], nargs="+")
    parser.add_argument('--n_shot', default=[0, 1, 2, 4, 8], nargs="+", help='Number of support images.')

    parser.add_argument('--max-new-tokens', default=15, type=int, help='Max new tokens for generation.')
    parser.add_argument('--task_description', default='nothing', type=str, choices=['nothing', 'concise', 'detailed'], help='Detailed level of task description.')
    parser.add_argument('--wo_img', action='store_true', help='whether to use images in the prompt.')
    parser.add_argument('--wo_query_img', action='store_true', help='whether to use images at all in the prompt.')
    parser.add_argument('--w_blank_img', action='store_true', help='whether to use blank images in the prompt.')
    parser.add_argument('--w_blank_query_img', action='store_true', help='whether to use blank images in the prompt.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    return parser.parse_args()


def eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, n_shot):
    data_path = args.dataDir
    results = []
    max_new_tokens = args.max_new_tokens

    for query in tqdm(query_meta):
        
        n_shot_support = ICL_utils.select_demonstration(support_meta, n_shot, args.dataset, query=query)

        if args.wo_img:
            predicted_answer = model_inference.ICL_I2T_inference_wo_img(args, engine, args.dataset, model, tokenizer, query, 
                                                                      n_shot_support, data_path, processor, max_new_tokens)
        elif args.wo_query_img:
            predicted_answer = model_inference.ICL_I2T_inference_wo_q_img(args, engine, args.dataset, model, tokenizer, query, 
                                                                      n_shot_support, data_path, processor, max_new_tokens)
        elif args.w_blank_img:
            predicted_answer = model_inference.ICL_I2T_inference_w_blank_img(args, engine, args.dataset, model, tokenizer, query, 
                                                                      n_shot_support, data_path, processor, max_new_tokens)
        elif args.w_blank_query_img:
            predicted_answer = model_inference.ICL_I2T_inference_w_blank_query_img(args, engine, args.dataset, model, tokenizer, query, 
                                                                      n_shot_support, data_path, processor, max_new_tokens)
        else:
            predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                      n_shot_support, data_path, processor, max_new_tokens)
        query['prediction'] = predicted_answer
        results.append(query)

    return results
    

if __name__ == "__main__":
    args = parse_args()

    query_meta, support_meta = utils.load_data(args)
    
    for engine in args.engine:

        model, tokenizer, processor = load_models.load_i2t_model(engine, args)
        print("Loaded model: {}\n".format(engine))
        
        utils.set_random_seed(args.seed)
        for shot in args.n_shot:
            results_dict = eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, int(shot))
            root_dir = os.path.dirname(os.path.abspath(__file__))
            if args.wo_img:
                results_dir = f"{root_dir}/results/wo_img"
            elif args.wo_query_img:
                results_dir = f"{root_dir}/results/wo_query_img"
            else:
                results_dir = f"{root_dir}/results/w_img"
            os.makedirs(f"{results_dir}/{args.dataset}", exist_ok=True)
            with open(f"{results_dir}/{args.dataset}/{engine}_{shot}-shot.json", "w") as f:
                json.dump(results_dict, f, indent=4)

        del model, tokenizer, processor
        torch.cuda.empty_cache()
        gc.collect()