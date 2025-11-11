import torch
try:
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import tokenizer_image_token
except:
    pass

import os
import time
from PIL import Image
from .ICL_utils import get_task_instruction, format_answer
from .utils import load_image, encode_image
from tqdm import tqdm

def generate_img_caption(engine, model,dataset, data_path, meta_data, processor, max_new_tokens=100):
    if dataset == 'open_mi':
        filenames = []
        for img_path, v in meta_data['mapping'].items():
            filenames.append(os.path.join(data_path, img_path))
        if "qwen2.5-vl" in engine:
            print(f"Generating captions for using Qwen2.5-VL...")
            from qwen_vl_utils import process_vision_info
            messages = []
            captions = []
            for filename in filenames:
                messages.append([{"role": "user",
                                  "content": [{"type": "image", "image": filename}, {"type": "text", "text": "Describe the image and list all the objects in this image."}]
                                  }])
            messages_chunks = [messages[i:i+10] for i in range(0, len(messages), 10)]
            for msg in tqdm(messages_chunks):
                text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(msg)
                inputs = processor(text=text, images=image_inputs, return_tensors="pt", padding=True, padding_side="left")
                inputs = inputs.to(model.device)
                with torch.no_grad():
                    pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
                pred_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, pred)]
                output_text = processor.batch_decode(pred_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                print(output_text)
                captions.extend(output_text)
            return captions
        
def generate_perception_classification(engine, dataset, model, tokenizer, 
                      all_shots, data_path, processor, max_new_tokens):
    if dataset == 'open_mi':
        classification_results = []
        for shot in tqdm(all_shots):
            query = shot['query']
            query_img = query['image']
            query_img_path = os.path.join(data_path, query_img[0])
            supports = shot['support']['5-shot']
            classes = query['classes']
            img_paths = []
            support_img_paths = []
            img_paths.append(query_img_path)
            for support in supports:
                support_img = support['image'][0]
                support_img_paths.append(support_img)
                support_img_path = os.path.join(data_path, support_img)
                img_paths.append(support_img_path)
            instruction = f"You are a helpful assistant. Classify the image into one of the following categories: {classes[0]}, {classes[1]}. Please answer directly with the category name."
            messages = []
            if "qwen2.5-vl" in engine:
                from qwen_vl_utils import process_vision_info
                for img_path in img_paths:
                    message = [
                        {"role": "user", 
                        "content": [
                            {"type": "text", 
                            "text": instruction },
                            {"type": "image", "image": img_path}
                                ]}]
                    messages.append(message)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                inputs = processor(text=text, images=image_inputs, return_tensors="pt", padding=True, padding_side="left")
                inputs = inputs.to(model.device)
                with torch.no_grad():
                    pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
                pred_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, pred)]
                output_text = processor.batch_decode(pred_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                predicted_answers = output_text
                print(predicted_answers)
                classification_results.append({
                    "query_image": query_img,
                    "query_real_name": query['real_name'],
                    "predicted_query_answer": predicted_answers[0],
                    "support_images": support_img_paths,
                    "support_real_names": [support['real_name'] for support in supports],
                    "predicted_support_answers": predicted_answers[1:],
                })
    return classification_results

def ICL_I2T_inference(args, engine, dataset, model, tokenizer, query, 
                      n_shot_support, data_path, processor, max_new_tokens, blank_demo_img=False, blank_query_img=False, rule_given=False, demo_desc=False, query_desc=False, demo_img_desc=False, query_img_desc=False, demo_img_desc_after_labels=False):
    blank_path = os.path.join(data_path, "blank.png")
    if not os.path.exists(blank_path):
        Image.new("RGB", (224, 224), (255, 255, 255)).save(blank_path)
    rule, task_instruction = get_task_instruction(args)
    img_id = query['image']
    query_images, query_image_paths = load_image(img_id, data_path)
    query_text = query['question']
    if 'qwen-vl' in engine:
        inputs = [{'text': f'You are a helpful assistant. {task_instruction}'}]
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                if blank_demo_img:
                    inputs.append({'image': blank_path})
                else:
                    inputs.append({'image': os.path.join(data_path, image_path)})
            inputs.append({'text': 'User: ' + n_shot_support[i]['question'] + 
                            '\nAssistant: ' + format_answer(n_shot_support[i]['answer'], dataset, query) + '\n'})
        
        for query_image_path in query_image_paths:
            if blank_query_img:
                inputs.append({'image': blank_path})
            else:
                inputs.append({'image': query_image_path})
        inputs.append({'text': 'User: ' + query_text + '\nAssistant:'})
        
        total_inputs = tokenizer.from_list_format(inputs)
        inputs = tokenizer(total_inputs, return_tensors='pt')
        inputs = inputs.to(model.device)
        with torch.no_grad():
            pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
        input_token_len = inputs['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(pred[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif "qwen2.5-vl" in engine:
        from qwen_vl_utils import process_vision_info
        if rule_given:
            if args.dataset == 'open_mi':
                mappings = []
                for i in range(len(n_shot_support)):
                    mapping = n_shot_support[i]['mapping']
                    mappings.append(mapping)
                mappings = ', '.join(mappings)
                rule = rule.format(mappings)
            instruction = f'You are a helpful assistant. {rule}'
            messages = [
            {"role": "system", "content": instruction},
            ]
        else:
            instruction = f'You are a helpful assistant. {task_instruction}'
            messages = [
            {"role": "system", "content": instruction},
            ]
            for i in range(len(n_shot_support)):
                for image_path in n_shot_support[i]['image']:
                    if blank_demo_img:
                        messages.append({"role":"user", "content": [{"type": "image", "image": blank_path}]})
                    elif demo_desc:
                        messages.append({"role":"user", "content": [{"type": "text", "text": f"This is an image of a {n_shot_support[i]['real_name']}."}]})
                    elif demo_img_desc:
                        messages.append({"role":"user", "content": [{"type": "image", "image": os.path.join(data_path, image_path)}]})
                        messages.append({"role":"user", "content": [{"type": "text", "text": f"This is an image of a {n_shot_support[i]['real_name']}."}]})
                    else:   
                        messages.append({"role":"user", "content": [{"type": "image", "image": os.path.join(data_path, image_path)}]})
                messages.append({"role":"user", "content": [{"type": "text", "text": n_shot_support[i]['question']}]})
                messages.append({"role":"assistant", "content": [{"type": "text", "text": format_answer(n_shot_support[i]['answer'], dataset, query)}]})
                if demo_img_desc_after_labels:
                    messages.append({"role":"user", "content": [{"type": "text", "text": f"This is a {n_shot_support[i]['real_name']}."}]})
        for query_image_path in query_image_paths:
            if blank_query_img:
                messages.append({"role":"user", "content": [{"type": "image", "image": blank_path}]})
            elif query_desc:
                messages.append({"role":"user", "content": [{"type": "text", "text": f"This is an image of a {query['real_name']}."}]})
            elif query_img_desc:
                messages.append({"role":"user", "content": [{"type": "image", "image": query_image_path}]})
                messages.append({"role":"user", "content": [{"type": "text", "text": f"This is an image of a {query['real_name']}."}]})
            else:
                messages.append({"role":"user", "content": [{"type": "image", "image": query_image_path}]})
        messages.append({"role":"user", "content": [{"type": "text", "text": query_text}]})
        print(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        with torch.no_grad():
            pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
        pred_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, pred)]
        output_text = processor.batch_decode(pred_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        predicted_answers = output_text[0]
        
    elif 'llava' in engine:
        from llava.conversation import conv_templates
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
        )
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
            input_text += f"{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"
        
        for query_image in query_images:
            images.append(query_image)
            input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
        input_text += f"{query_text}\nAnswer:"
        image_tensor = torch.stack(
                [
                    processor.preprocess(image_file, return_tensors="pt")["pixel_values"][0]
                    for image_file in images
                ]
            )
        image_tensor = image_tensor.half().cuda()
        conv_mode = 'llava_v1' if 'onevision' not in engine else 'qwen_1_5'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                )
        predicted_answers = tokenizer.batch_decode(generated_ids[:, :], skip_special_tokens=True)[0]
    elif 'flamingo' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                if blank_demo_img:
                    images.append(Image.open(blank_path).convert("RGB"))
                else:
                    images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += "<image>"
            input_text += f"{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}<|endofchunk|>"
        for query_image in query_images:
            if blank_query_img:
                images.append(Image.open(blank_path).convert("RGB"))
            else:
                images.append(query_image)
            input_text += "<image>"
            
        vision_x = [processor(image).unsqueeze(0) for image in images]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

        input_text += f"{query_text}\nAnswer:"
        
        lang_x = tokenizer(
            [input_text],
            return_tensors="pt",
        )
        with torch.no_grad():
            predicted_answers = model.generate(
                vision_x=vision_x.to(torch.bfloat16).cuda(),
                lang_x=lang_x["input_ids"].cuda(),
                attention_mask=lang_x["attention_mask"].cuda(),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        input_token_len = lang_x['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(predicted_answers[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif 'otter' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                if blank_demo_img:
                    images.append(Image.open(blank_path).convert("RGB"))
                else:
                    images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += "<image>"
            input_text += f"User: {n_shot_support[i]['question']}\nGPT:<answer> {format_answer(n_shot_support[i]['answer'], dataset, query)}<|endofchunk|>"
        for query_image in query_images:
            if blank_query_img:
                images.append(Image.open(blank_path).convert("RGB"))
            else:
                images.append(query_image)
            input_text += "<image>"
        input_text += f"User: {query_text}\nGPT:<answer>"

        vision_x = processor.preprocess(images, return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        lang_x = model.text_tokenizer(
            [
                input_text,
            ],
            return_tensors="pt",
        )
        bad_words_id = tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
        with torch.no_grad():
            predicted_answers = model.generate(
                vision_x=vision_x.to(model.device),
                lang_x=lang_x["input_ids"].to(model.device),
                attention_mask=lang_x["attention_mask"].to(model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                bad_words_ids=bad_words_id,
            )
        input_token_len = lang_x['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(predicted_answers[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif 'internlm-x' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                if blank_demo_img:
                    image = Image.open(blank_path).convert("RGB")
                else:
                    image = Image.open(os.path.join(data_path, image_path)).convert("RGB")
                image = model.vis_processor(image)
                images.append(image)
                input_text += "<ImageHere>"
            input_text += f"{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"
        for query_image in query_images:
            if blank_query_img:
                images.append(model.vis_processor(Image.open(blank_path).convert("RGB")))
            else:
                images.append(model.vis_processor(query_image))
            input_text += "<ImageHere>"
        input_text += f"{query_text}\nAnswer:"
        image = torch.stack(images).to(torch.bfloat16).cuda()
        predicted_answers, history = model.chat(tokenizer, query=input_text, image=image, history=[], do_sample=False, max_new_tokens=max_new_tokens)
    elif 'emu2-chat' in engine:
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                if blank_demo_img:
                    images.append(Image.open(blank_path).convert("RGB"))
                else:
                    images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                input_text += "[<IMG_PLH>]"
            input_text += f"[{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}]."
        for query_image in query_images:
            if blank_query_img:
                images.append(Image.open(blank_path).convert("RGB"))
            else:
                images.append(query_image)
            input_text += "[<IMG_PLH>]"
        input_text += f"[{query_text}\nAnswer:"
        inputs = model.build_input_ids(
            text=[input_text],
            tokenizer=tokenizer,
            image=images
        )
        
        with torch.no_grad():
            predicted_answers = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16),
                max_new_tokens=max_new_tokens,)
        predicted_answers = tokenizer.decode(predicted_answers[:, :].cpu()[0], skip_special_tokens=True)
        
    elif 'idefics' in engine:
        prompts = [f"You are a helpful assistant.\n{task_instruction}\n"]
        for i in range(len(n_shot_support)):
            for image_path in n_shot_support[i]['image']:
                if blank_demo_img:
                    prompts.append(Image.open(blank_path).convert("RGB"))
                else:
                    prompts.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
            prompts.append(f"\nUser: {n_shot_support[i]['question']}")
            #prompts.append("<end_of_utterance>")
            prompts.append(f"\nAssistant: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n")
        for query_image in query_images:
            if blank_query_img:
                prompts.append(Image.open(blank_path).convert("RGB"))
            else:
                prompts.append(query_image)
        prompts.append(f"\nUser: {query_text}")
        #prompts.append("<end_of_utterance>")
        prompts.append("\nAssistant:")
        inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to("cuda")
        exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

        generated_ids = model.generate(**inputs, 
                                       eos_token_id=exit_condition, 
                                       bad_words_ids=bad_words_ids, 
                                       max_new_tokens=max_new_tokens,
                                       do_sample=False)
        input_token_len = inputs['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(generated_ids[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif 'gpt4v' in engine:
        import openai
        from openai import OpenAI
        # configure your openai key by `export OPENAI_API_KEY=""` in command line
        api_key = os.environ['OPENAI_API_KEY']
        client = OpenAI(api_key=api_key)
        rule, task_instruction = get_task_instruction(args)
        img_id = query['image']
        query_images, query_image_paths = load_image(img_id, data_path)
        query_text = query['question']
        
        content = [{
                "type": "text",
                "text": f"{task_instruction}\nEnsure the generated answers only contain the answer to the question and no other information."
            }]
        for item in n_shot_support:
            for image_path in item['image']:
                if blank_demo_img:
                    base64_image, mime_type = encode_image(blank_path)
                else:
                    base64_image, mime_type = encode_image(os.path.join(data_path, image_path))
                content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "low"},
                })
            content.append({
                    "type": "text",
                    "text": item['question']
            })
            content.append({
                    "type": "text",
                    "text": "The answer is " + str(item['answer'])
            })
        for query_image_path in query_image_paths:
            if blank_query_img:
                base64_image, mime_type = encode_image(blank_path)
            else:
                base64_image, mime_type = encode_image(os.path.join(data_path, query_image_path))
            content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}",
                                  "detail": "low"},
                    
            })
        content.append({
                "type": "text",
                "text": query_text + " The answer is"
        })
        messages = [{
            "role": "user",
            "content": content
        }]
        while True:
            try:
                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=messages,
                    max_tokens=max_new_tokens,
                )
                predicted_answers = response.choices[0].message.content
                print(query['id'], '\t', predicted_answers)
                break
            except openai.RateLimitError as e:
                print("Rate limit reached, waiting for 1 hour")
                time.sleep(3600)  # Wait for 1 hour (3600 seconds)
                continue
            except Exception as e:
                print("pausing")
                time.sleep(1)
                continue
    elif 'phi-3' in engine:
        images = []
        image_placeholders = ""
        full_text_prompt = f"{task_instruction}\n"
        img_idx = 1
        for shot in n_shot_support:
            for image_path in shot['image']:
                if blank_demo_img:
                    images.append(Image.open(blank_path).convert("RGB"))
                else:
                    images.append(Image.open(os.path.join(data_path, image_path)).convert("RGB"))
                image_placeholders = f"<|image_{img_idx}|>\n"
                full_text_prompt += image_placeholders
                img_idx += 1
            full_text_prompt += f"{shot['question']}\nAnswer: {format_answer(shot['answer'], dataset, query)}\n"
        for query_image in query_images:
            if blank_query_img:
                images.append(Image.open(blank_path).convert("RGB"))
            else:
                images.append(query_image)
            image_placeholders = f"<|image_{img_idx}|>\n"
            full_text_prompt += image_placeholders
            img_idx += 1
        full_text_prompt += f"{query_text}\nAnswer:"
        messages = [{'role': 'user', 'content': full_text_prompt}]
        prompt =  processor.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
        inputs = processor(prompt, images, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
        predicted_answers = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    return predicted_answers

# def ICL_I2T_inference_w_blank_img(args, engine, dataset, model, tokenizer, query, 
#                       n_shot_support, data_path, processor, max_new_tokens, blank_demo_img=False, blank_query_img=False):
#     blank_path = os.path.join(data_path, "blank.png")
#     if not os.path.exists(blank_path):
#         Image.new("RGB", (224, 224), (255, 255, 255)).save(blank_path)
#     rule, task_instruction = get_task_instruction(args)
#     img_id = query['image']
#     query_images, query_image_paths = load_image(img_id, data_path)
#     query_text = query['question']
#     if 'qwen-vl' in engine:
#         inputs = [{'text': f'You are a helpful assistant. {task_instruction}'}]
#         for i in range(len(n_shot_support)):
#             for image_path in n_shot_support[i]['image']:
#                 inputs.append({'image': blank_path})
#             inputs.append({'text': 'User: ' + n_shot_support[i]['question'] + 
#                             '\nAssistant: ' + format_answer(n_shot_support[i]['answer'], dataset, query) + '\n'})
        
#         for query_image_path in query_image_paths:
#             inputs.append({'image': query_image_path})
#         inputs.append({'text': 'User: ' + query_text + '\nAssistant:'})
        
#         total_inputs = tokenizer.from_list_format(inputs)
#         inputs = tokenizer(total_inputs, return_tensors='pt')
#         inputs = inputs.to(model.device)
#         with torch.no_grad():
#             pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
#         input_token_len = inputs['input_ids'].shape[1]
#         predicted_answers = tokenizer.decode(pred[:, input_token_len:].cpu()[0], skip_special_tokens=True)
#     elif "qwen2.5-vl" in engine:
#         from qwen_vl_utils import process_vision_info
#         instruction = f'You are a helpful assistant. {task_instruction}'
#         messages = [
#            {"role": "system", "content": instruction},
#            {"role": "user", "content": []},
#         ]
#         for i in range(len(n_shot_support)):
#             for image_path in n_shot_support[i]['image']:
#                 if blank_demo_img:
#                     messages[-1]['content'].append({"type": "image", "image": blank_path})
#                 else:
#                     messages[-1]['content'].append({"type": "image", "image": image_path})
#             messages[-1]['content'].append({"type": "text", "text": n_shot_support[i]['question']})
#             messages[-1]['content'].append({"type": "text", "text": format_answer(n_shot_support[i]['answer'], dataset, query)})
#         for query_image_path in query_image_paths:
#             if blank_query_img:
#                 messages[-1]['content'].append({"type": "image", "image": blank_path})
#             else:
#                 messages[-1]['content'].append({"type": "image", "image": query_image_path})
#         messages[-1]['content'].append({"type": "text", "text": query_text})
#         text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         image_inputs, _ = process_vision_info(messages)
#         inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
#         inputs = inputs.to(model.device)
#         with torch.no_grad():
#             pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
#         pred_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, pred)]
#         output_text = processor.batch_decode(pred_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#         predicted_answers = output_text[0]
#     return predicted_answers

# def ICL_I2T_inference_w_blank_query_img(args, engine, dataset, model, tokenizer, query, 
#                       n_shot_support, data_path, processor, max_new_tokens):
#     blank_path = os.path.join(data_path, "blank.png")
#     if not os.path.exists(blank_path):
#         Image.new("RGB", (224, 224), (255, 255, 255)).save(blank_path)
#     rule, task_instruction = get_task_instruction(args)
#     img_id = query['image']
#     query_images, query_image_paths = load_image(img_id, data_path)
#     query_text = query['question']
#     if 'qwen-vl' in engine:
#         inputs = [{'text': f'You are a helpful assistant. {task_instruction}'}]
#         for i in range(len(n_shot_support)):
#             for image_path in n_shot_support[i]['image']:
#                 inputs.append({'image': blank_path})
#             inputs.append({'text': 'User: ' + n_shot_support[i]['question'] + 
#                             '\nAssistant: ' + format_answer(n_shot_support[i]['answer'], dataset, query) + '\n'})
        
#         for query_image_path in query_image_paths:
#             inputs.append({'image': blank_path})
#         inputs.append({'text': 'User: ' + query_text + '\nAssistant:'})
        
#         total_inputs = tokenizer.from_list_format(inputs)
#         inputs = tokenizer(total_inputs, return_tensors='pt')
#         inputs = inputs.to(model.device)
#         with torch.no_grad():
#             pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
#         input_token_len = inputs['input_ids'].shape[1]
#         predicted_answers = tokenizer.decode(pred[:, input_token_len:].cpu()[0], skip_special_tokens=True)
#     elif "qwen2.5-vl" in engine:
#         from qwen_vl_utils import process_vision_info
#         instruction = f'You are a helpful assistant. {task_instruction}'
#         messages = [
#            {"role": "system", "content": instruction},
#            {"role": "user", "content": []},
#         ]
#         for i in range(len(n_shot_support)):
#             for image_path in n_shot_support[i]['image']:
#                 messages[-1]['content'].append({"type": "image", "image": blank_path})
#             messages[-1]['content'].append({"type": "text", "text": n_shot_support[i]['question']})
#             messages[-1]['content'].append({"type": "text", "text": format_answer(n_shot_support[i]['answer'], dataset, query)})
#         for query_image_path in query_image_paths:
#             messages[-1]['content'].append({"type": "image", "image": blank_path})
#         messages[-1]['content'].append({"type": "text", "text": query_text})
#         text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         image_inputs, _ = process_vision_info(messages)
#         inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
#         inputs = inputs.to(model.device)
#         with torch.no_grad():
#             pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
#         pred_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, pred)]
#         output_text = processor.batch_decode(pred_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#         predicted_answers = output_text[0]
#     return predicted_answers
    
    
def ICL_I2T_inference_wo_img(args, engine, dataset, model, tokenizer, query, 
                      n_shot_support, data_path, processor, max_new_tokens):
    rule, task_instruction = get_task_instruction(args)
    img_id = query['image']
    query_images, query_image_paths = load_image(img_id, data_path)
    query_text = query['question']
    if 'qwen-vl' in engine:
        inputs = [{'text': f'You are a helpful assistant. {task_instruction}'}]
        for i in range(len(n_shot_support)):
            inputs.append({'text': 'User: ' + n_shot_support[i]['question'] + 
                            '\nAssistant: ' + format_answer(n_shot_support[i]['answer'], dataset, query) + '\n'})
        for query_image_path in query_image_paths:
            inputs.append({'image': query_image_path})
        inputs.append({'text': 'User: ' + query_text + '\nAssistant:'})
        
        total_inputs = tokenizer.from_list_format(inputs)
        inputs = tokenizer(total_inputs, return_tensors='pt')
        inputs = inputs.to(model.device)
        with torch.no_grad():
            pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
        input_token_len = inputs['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(pred[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif "qwen2.5-vl" in engine:
        from qwen_vl_utils import process_vision_info
        instruction = f'You are a helpful assistant. {task_instruction}'
        messages = [
           {"role": "system", "content": instruction},
           {"role": "user", "content": []},
        ]
        for i in range(len(n_shot_support)):
            messages[-1]['content'].append({"type": "text", "text": n_shot_support[i]['question']})
            messages[-1]['content'].append({"type": "text", "text": format_answer(n_shot_support[i]['answer'], dataset, query)})
        for query_image_path in query_image_paths:
            messages[-1]['content'].append({"type": "image", "image": query_image_path})
        messages[-1]['content'].append({"type": "text", "text": query_text})
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        with torch.no_grad():
            pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
        pred_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, pred)]
        output_text = processor.batch_decode(pred_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        predicted_answers = output_text[0]
        
    elif 'phi-3' in engine:
        images = []
        img_idx = 1
        full_text_prompt = f"{task_instruction}\n"
        for shot in n_shot_support:
            full_text_prompt += f"{shot['question']}\nAnswer: {format_answer(shot['answer'], dataset, query)}\n"
        for query_image in query_images:
            images.append(query_image)
            image_placeholders = f"<|image_{img_idx}|>\n"
            full_text_prompt += image_placeholders
            img_idx += 1
        full_text_prompt += f"{query_text}\nAnswer:"
        messages = [{'role': 'user', 'content': full_text_prompt}]
        prompt =  processor.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
        inputs = processor(prompt, images, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
        predicted_answers = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    elif 'llava' in engine:
        from llava.conversation import conv_templates
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
        )
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        images = []
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            input_text += f"{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"
        
        for query_image in query_images:
            images.append(query_image)
            input_text += f"{DEFAULT_IMAGE_TOKEN}\n"
        input_text += f"{query_text}\nAnswer:"
        image_tensor = torch.stack(
                [
                    processor.preprocess(image_file, return_tensors="pt")["pixel_values"][0]
                    for image_file in images
                ]
            )
        image_tensor = image_tensor.half().cuda()
        conv_mode = 'llava_v1' if 'onevision' not in engine else 'qwen_1_5'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                )
        predicted_answers = tokenizer.batch_decode(generated_ids[:, :], skip_special_tokens=True)[0]
    return predicted_answers

def ICL_I2T_inference_wo_q_img(args, engine, dataset, model, tokenizer, query, 
                      n_shot_support, data_path, processor, max_new_tokens):
    rule, task_instruction = get_task_instruction(args)
    query_text = query['question']
    if 'qwen-vl' in engine:
        inputs = [{'text': f'You are a helpful assistant. {task_instruction}'}]
        for i in range(len(n_shot_support)):
            inputs.append({'text': 'User: ' + n_shot_support[i]['question'] + 
                            '\nAssistant: ' + format_answer(n_shot_support[i]['answer'], dataset, query) + '\n'})
        inputs.append({'text': 'User: ' + query_text + '\nAssistant:'})
        
        total_inputs = tokenizer.from_list_format(inputs)
        inputs = tokenizer(total_inputs, return_tensors='pt')
        inputs = inputs.to(model.device)
        with torch.no_grad():
            pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
        input_token_len = inputs['input_ids'].shape[1]
        predicted_answers = tokenizer.decode(pred[:, input_token_len:].cpu()[0], skip_special_tokens=True)
    elif "qwen2.5-vl" in engine:
        from qwen_vl_utils import process_vision_info
        instruction = f'You are a helpful assistant. {task_instruction}'
        messages = [
           {"role": "system", "content": instruction},
           {"role": "user", "content": []},
        ]
        for i in range(len(n_shot_support)):
            messages[-1]['content'].append({"type": "text", "text": n_shot_support[i]['question']})
            messages[-1]['content'].append({"type": "text", "text": format_answer(n_shot_support[i]['answer'], dataset, query)})
        messages[-1]['content'].append({"type": "text", "text": query_text})
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        with torch.no_grad():
            pred = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, min_new_tokens=1)
        pred_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, pred)]
        output_text = processor.batch_decode(pred_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        predicted_answers = output_text[0]
    elif 'phi-3' in engine:
        full_text_prompt = f"{task_instruction}\n"
        for shot in n_shot_support:
            full_text_prompt += f"{shot['question']}\nAnswer: {format_answer(shot['answer'], dataset, query)}\n"
        full_text_prompt += f"{query_text}\nAnswer:"
        messages = [{'role': 'user', 'content': full_text_prompt}]
        prompt =  processor.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
        inputs = processor(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
        predicted_answers = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    elif 'llava' in engine:
        from llava.conversation import conv_templates
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
        )
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        input_text = f"{task_instruction}\n"
        for i in range(len(n_shot_support)):
            input_text += f"{n_shot_support[i]['question']}\nAnswer: {format_answer(n_shot_support[i]['answer'], dataset, query)}\n"
        input_text += f"{query_text}\nAnswer:"
        image_tensor = None
        conv_mode = 'llava_v1' if 'onevision' not in engine else 'qwen_1_5'
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                )
        predicted_answers = tokenizer.batch_decode(generated_ids[:, :], skip_special_tokens=True)[0]
    return predicted_answers

