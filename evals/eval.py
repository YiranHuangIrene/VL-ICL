import numpy as np
import re
import torch
import os
from tqdm import tqdm

def eval_perception_contain(results, dataset, meta_data):
    # results: a json file contains the results for perception tasks
    # dataset: the dataset name
    # meta_data: the meta data of the dataset,
    if dataset == 'open_mi':
        acc = []
        failure_cases = []
        i = 0
        synonyms = meta_data['synonyms']
        for img, label in meta_data['mapping'].items():
            prediction = str(results[i]).lower()
            i += 1
            label_synonyms = synonyms[label]
            # if any of the synonyms is in the prediction, then the prediction is correct. If the labek comprises of multiple words, then the prediction is correct if both of the words in the label is in the prediction but not necessarily in the same order.
            for synonym in label_synonyms:
                if synonym.lower() in prediction or all(word.lower() in prediction for word in synonym.split()):
                    acc.append(1)
                    break
            else:
                failure_cases.append([img, label, prediction])
                acc.append(0)
        avg_acc = np.average(acc) * 100.0
    return avg_acc, failure_cases

def eval_perception_llm_judge(results, dataset, meta_data):
    # LLM as judge
    if dataset == 'open_mi':
        acc = []
        output_dict = []
        i = 0
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16,attn_implementation="flash_attention_2",).eval().to("cuda")
        for img, label in tqdm(meta_data['mapping'].items()):
            prediction = str(results[i]).lower()
            i += 1 
            messages = [
                {"role": "user", "content": f"You are a cautious object-presence judge. Decide whether {label} is present based on the provided image caption. Answer directly with 'yes' or 'no'. Image caption: {prediction}"},
            ]
            text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
                )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # conduct text completion
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=5,
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            # print(content)
            output_dict.append(
                {'image': img,
                 'label': label,
                 'prediction': prediction,
                 'llm_judge': content}
            )
            if 'yes' in content.lower():
                acc.append(1)
            else:
                acc.append(0)
        avg_acc = np.average(acc) * 100.0
        output = {"accuracy": avg_acc, "perception": output_dict}
    return avg_acc, output
        
def eval_perception_classification(results, dataset):
    if dataset == 'open_mi':
        acc_query = 0
        acc_support = 0
        acc_all = 0
        query_labels = []
        query_predictions = []
        support_labels = []
        support_predictions = []
        results_dict = {}
        for result in results:
            query_labels.append(result["query_real_name"].lower())
            query_predictions.append(result["predicted_query_answer"].lower())
        avg_acc_query = sum(l == p for l, p in zip(query_labels, query_predictions)) / len(query_labels)
        results_dict["avg_acc_query"] = avg_acc_query * 100.0
        for n_shot in [1,2,4,5]:
            for result in results:
                data_real_names = result[f"support_real_names"][: (2 * n_shot)]
                data_predictions = result[f"predicted_support_answers"][: (2 * n_shot)]
                support_labels.extend([label.lower() for label in data_real_names])
                support_predictions.extend([answer.lower() for answer in data_predictions])
                assert len(support_labels) == len(support_predictions)
            avg_acc_support = sum(l == p for l, p in zip(support_labels, support_predictions)) / len(support_labels) * 100.0
            all_labels = support_labels + query_labels
            all_preds = support_predictions + query_predictions
            avg_acc_all = sum(l == p for l, p in zip(all_labels, all_preds)) / len(all_labels) * 100.0
            results_dict[f"avg_acc_support_{n_shot}"] = avg_acc_support
            results_dict[f"avg_acc_all_{n_shot}"] = avg_acc_all
        return results_dict
            

def eval_scores(results, dataset, model=None, tokenizer=None, processor=None):
    if dataset in ['textocr', 'operator_induction', 'clevr', 'open_mi',
                    'operator_induction_interleaved']:
        score = exact_match(results, dataset)
    elif dataset == 'matching_mi':
        score = exact_yes_no(results)
    elif dataset == 'open_t2i_mi' or dataset == 'operator_induction_t2i' or dataset == 'fast_attr_t2i' or dataset == 'fast_count_t2i':
        score = llava_judge_t2i(results, model, tokenizer, processor, dataset)
    elif dataset == 'cobsat':
        score = llava_judge_cobsat(results, model, tokenizer, processor)
    return score

def eval_scores_contain(results, dataset, model=None, tokenizer=None, processor=None):
    if dataset in [ 'clevr']:
        score = exact_in_match(results, dataset)
    elif dataset in ['textocr','open_mi','operator_induction', 'operator_induction_interleaved']:
        score = exact_match(results, dataset)
    elif dataset == 'matching_mi':
        score = exact_yes_no(results)
    elif dataset == 'open_t2i_mi' or dataset == 'operator_induction_t2i' or dataset == 'fast_attr_t2i' or dataset == 'fast_count_t2i':
        score = llava_judge_t2i(results, model, tokenizer, processor, dataset)
    elif dataset == 'cobsat':
        score = llava_judge_cobsat(results, model, tokenizer, processor)
    return score

def exact_yes_no(results):
    acc = []
    for result in results:
        prediction = result['prediction'].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if result['answer'].lower() == 'yes' and 'yes' in str(prediction).lower():
            acc.append(1)
        elif result['answer'].lower() == 'no' and 'yes' not in str(prediction).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc

def exact_in_match(results, dataset):
    acc = []
    for result in results:
        prediction = result['prediction'].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if 'operator_induction' in dataset or 'clevr_simple' in dataset:
            # find the number
            match = re.search(r'\d+', prediction)
            if match:
                prediction = match.group()
            else:
                prediction = ''
        if str(result['answer']).lower() in str(prediction).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc

def exact_match(results, dataset):
    acc = []
    for result in results:
        prediction = result['prediction'].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if 'operator_induction' in dataset or 'clevr_simple' in dataset:
            # find the number
            match = re.search(r'\d+', prediction)
            if match:
                prediction = match.group()
            else:
                prediction = ''

        if str(prediction).lower() == str(result['answer']).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc

def llava_judge_open_t2i_mi(results, model, tokenizer, processor):
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.mm_utils import tokenizer_image_token
    from PIL import Image

    acc = []
    for result in results:
        prompt = f"Decide whether the image contains {result['answer']}. Answer with 'yes' or 'no'.\n"
        image = Image.open(result['prediction']).convert('RGB')
        image_tensor =  processor.preprocess([image], return_tensors='pt')['pixel_values'].cuda().half()
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        
        conv_mode = 'llava_v1'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0),
                do_sample=False,
                temperature=1,
                max_new_tokens=5,
                min_new_tokens=1,
                )
            
        input_token_len = input_ids.shape[1]
        predicted_answers = tokenizer.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)[0]
        if 'yes' in predicted_answers.lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc

def llava_judge_cobsat(results, model, tokenizer, processor):
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.mm_utils import tokenizer_image_token
    from PIL import Image

    acc = []
    acc_latent = []
    acc_non_latent = []
    for result in results:
        accs = []
        for answer in [result['answer'][0], result['answer'][1]]:
            prompt = f"Decide whether the image contains the following concept: {answer}. Answer with 'yes' or 'no'.\n"
            image = Image.open(result['prediction']).convert('RGB')
            image_tensor =  processor.preprocess([image], return_tensors='pt')['pixel_values'].cuda().half()
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            
            conv_mode = 'llava_v1'
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0),
                    do_sample=False,
                    temperature=1,
                    max_new_tokens=5,
                    min_new_tokens=1,
                    )
                
            input_token_len = input_ids.shape[1]
            predicted_answers = tokenizer.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)[0]
            if 'yes' in predicted_answers.lower():
                accs.append(1)
            else:
                accs.append(0)
        acc_latent.append(accs[0])
        acc_non_latent.append(accs[1])
        if accs[0] == 1 and accs[1] == 1:
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    avg_acc_latent = np.average(acc_latent)
    avg_acc_non_latent = np.average(acc_non_latent)
    return {'total': avg_acc, 'latent': avg_acc_latent, 'non_latent': avg_acc_non_latent}
