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
                                                   'llava-onevision-0.5b','phi-3.5-vision','phi-3-vision'], help='now only support qwen2.5-vl-3b, qwen2.5-vl-7b',
                        default=["llava16-7b"], nargs="+")
    parser.add_argument('--n_shot', default=[0, 1, 2, 4, 8], nargs="+", help='Number of support images.')

    parser.add_argument('--max-new-tokens', default=100, type=int, help='Max new tokens for generation. Normally, it should be 10. For perception tasks, it can be 100.')
    parser.add_argument('--task_description', default='nothing', type=str, choices=['nothing', 'concise', 'detailed', "rule"], help='Detailed level of task description.')
    parser.add_argument('--rule_only', action='store_true', help='Replace the shots with the rule -> Rule + query (image + question)')
    parser.add_argument('--rule_only_query_desc', action='store_true', help='Replace the shots with the rule, replace the query image with its description -> Rule + query (description + question)')
    parser.add_argument('--rule_demo', action='store_true', help='Complement the shots with the rule-style instruction -> Rule-style instruction + shots(images + descriptions + answers) + query (image + question)')
    parser.add_argument('--w_img', action='store_true', help='Baseline: Shots(images + question + answer) + query (image + question)')
    parser.add_argument('--w_blank_demo_img', action='store_true', help='Use blank images for the demo images. -> shots (blank images + question + answer) + query (image + question)')
    parser.add_argument('--w_blank_query_img', action='store_true', help='Use blank images for the query image. -> shots (demo images + question + answer) + query (blank image + question)')
    parser.add_argument('--w_blank_img_all', action='store_true', help='Use blank images for both the demo images and the query image. -> shots (blank images + question + answer) + query (blank image + question)')
    parser.add_argument('--demo_img_desc_query_img', action='store_true', help='Complement the demo images with their descriptions -> Shots(image + description + question + answer) + query (image + question)')
    parser.add_argument('--demo_img_desc_query_img_desc', action='store_true', help='Complement the demo images with their descriptions, complement the query image with its description -> Shots(image + description + question + answer) + query (image + description + question)')
    parser.add_argument('--demo_img_desc_query_img_desc_after_labels', action='store_true', help='Complement the demo images with their descriptions, complement the query image with its description, place the descriptions after the labels -> Shots(image + question + answer + description) + query (image + description + question)')
    parser.add_argument('--demo_desc_query_img', action='store_true', help='Replace the demo images with their descriptions -> Shots(descriptions + question + answer) + query (image + question)')
    parser.add_argument('--demo_desc_query_desc', action='store_true', help='Replace the demo images with their descriptions, replace the query image with its description -> Shots(descriptions + question + answer) + query (description + question)')
    parser.add_argument('--perception', action='store_true', help='Generate image description for both query and support images.')
    parser.add_argument('--perception_classification', action='store_true', help='Generate image classification for both query and support images.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    return parser.parse_args()

def generate_img_caption(args,engine,model, processor, meta_data):
    print(f"Generating captions for {len(meta_data)} images...")
    captions = model_inference.generate_img_caption(engine=engine, model=model, dataset=args.dataset, data_path=args.dataDir, meta_data=meta_data, processor=processor, max_new_tokens=args.max_new_tokens)
    return captions

def generate_img_classification(engine, dataset, model, tokenizer, 
                      all_shots, data_path, processor, max_new_tokens):
    print(f"Generating classifications for {len(meta_data)} images...")
    classifications = model_inference.generate_perception_classification(engine, dataset, model, tokenizer, 
                      all_shots, data_path, processor, max_new_tokens)
    return classifications

def eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, n_shot):
    data_path = args.dataDir
    results = []
    max_new_tokens = args.max_new_tokens
    output = {}
    for query in tqdm(query_meta):
        
        n_shot_support = ICL_utils.select_demonstration(support_meta, n_shot, args.dataset, query=query)

        if args.rule_only:
            predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                                n_shot_support, data_path, processor, max_new_tokens, rule_given=True)
        elif args.rule_only_query_desc:
            predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                                n_shot_support, data_path, processor, max_new_tokens, rule_given=True, query_desc=True)
        elif args.rule_demo:
            predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                                n_shot_support, data_path, processor, max_new_tokens)
        elif args.w_blank_demo_img:
            predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                                    n_shot_support, data_path, processor, max_new_tokens, blank_demo_img=True)
        elif args.w_blank_query_img:
            predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                            n_shot_support, data_path, processor, max_new_tokens, blank_query_img=True)
        elif args.w_blank_img_all:
            predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                            n_shot_support, data_path, processor, max_new_tokens, blank_demo_img=True, blank_query_img=True)
        elif args.demo_desc_query_img:
            predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                            n_shot_support, data_path, processor, max_new_tokens, demo_desc=True, query_desc=False)
        elif args.demo_desc_query_desc:
                predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                            n_shot_support, data_path, processor, max_new_tokens, demo_desc=True, query_desc=True)
        elif args.demo_img_desc_query_img:
            predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                            n_shot_support, data_path, processor, max_new_tokens, demo_img_desc=True, query_img_desc=False)  
        elif args.demo_img_desc_query_img_desc:
            predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                            n_shot_support, data_path, processor, max_new_tokens, demo_img_desc=True, query_img_desc=True)   
        elif args.demo_img_desc_query_img_desc_after_labels:
            predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                            n_shot_support, data_path, processor, max_new_tokens, demo_img_desc_after_labels=True, query_img_desc=True)
        else:
            assert args.w_img, "Please specify the prompt type."
            predicted_answer = model_inference.ICL_I2T_inference(args, engine, args.dataset, model, tokenizer, query, 
                                                    n_shot_support, data_path, processor, max_new_tokens)
        # for entry in ['id', 'image', 'question', 'answer', 'real_name']:
        #     output[entry] = query[entry]
        # output['prediction'] = predicted_answer
        # output['support'] = n_shot_support
        query['prediction'] = predicted_answer
        
        results.append(query)

    return results
    

if __name__ == "__main__":
    args = parse_args()

    query_meta, support_meta, meta_data = utils.load_data(args)
    
    for engine in args.engine:
        model, tokenizer, processor = load_models.load_i2t_model(engine, args)
        print("Loaded model: {}\n".format(engine))
        utils.set_random_seed(args.seed)
        if args.perception:
            results_dict = generate_img_caption(args, engine, model, processor,meta_data)
            root_dir = os.path.dirname(os.path.abspath(__file__))
            results_dir = f"{root_dir}/results/perception"
            os.makedirs(f"{results_dir}/{args.dataset}/{engine}", exist_ok=True)
            with open(f"{results_dir}/{args.dataset}/{engine}/perception.json", "w") as f:
                json.dump(results_dict, f, indent=4)
            print(f"Results saved to {results_dir}/{args.dataset}/{engine}/perception.json")
        elif args.perception_classification:
            results_dict = generate_img_classification(engine, args.dataset, model, tokenizer, meta_data, args.dataDir, processor, args.max_new_tokens)
            root_dir = os.path.dirname(os.path.abspath(__file__))
            results_dir = f"{root_dir}/results/perception_classification"
            os.makedirs(f"{results_dir}/{args.dataset}/{engine}", exist_ok=True)
            with open(f"{results_dir}/{args.dataset}/{engine}/perception_classification.json", "w") as f:
                json.dump(results_dict, f, indent=4)
            print(f"Results saved to {results_dir}/{args.dataset}/{engine}/perception_classification.json")
        else:
            for shot in args.n_shot:
                results_dict = eval_questions(args, query_meta, support_meta, model, tokenizer, processor, engine, int(shot))
                root_dir = os.path.dirname(os.path.abspath(__file__))
                if args.rule_only:
                    results_dir = f"{root_dir}/results/rule_only"
                elif args.rule_only_query_desc:
                    results_dir = f"{root_dir}/results/rule_only_query_desc"
                elif args.rule_demo:
                    results_dir = f"{root_dir}/results/rule_demo"
                elif args.w_blank_demo_img:
                    results_dir = f"{root_dir}/results/w_blank_demo_img"
                elif args.w_blank_query_img:
                    results_dir = f"{root_dir}/results/w_blank_query_img"
                elif args.w_blank_img_all:
                    results_dir = f"{root_dir}/results/w_blank_img_all"
                elif args.demo_img_desc_query_img:
                    results_dir = f"{root_dir}/results/demo_img_desc_query_img"
                elif args.demo_img_desc_query_img_desc:
                    results_dir = f"{root_dir}/results/demo_img_desc_query_img_desc"
                elif args.demo_img_desc_query_img_desc_after_labels:
                    results_dir = f"{root_dir}/results/demo_img_desc_query_img_desc_after_labels"
                elif args.demo_desc_query_img:
                    results_dir = f"{root_dir}/results/demo_desc_query_img"
                elif args.demo_desc_query_desc:
                    results_dir = f"{root_dir}/results/demo_desc_query_desc"
                else:
                    assert args.w_img, "Please specify the prompt type."
                    results_dir = f"{root_dir}/results/w_img"
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                os.makedirs(f"{results_dir}/{args.dataset}/{engine}", exist_ok=True)
                with open(f"{results_dir}/{args.dataset}/{engine}/{shot}-shot.json", "w") as f:
                    json.dump(results_dict, f, indent=4)
                print(f"Results saved to {results_dir}/{args.dataset}/{engine}/{shot}-shot.json")

        del model, tokenizer, processor
        torch.cuda.empty_cache()
        gc.collect()