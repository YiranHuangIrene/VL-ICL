import random
import copy


def select_demonstration(support_meta, n_shot, dataset, query=None):
    if 'operator_induction' in dataset:
        operator_index = {'+': 0, '-': 1, 'x': 2}
        n_shot_support_raw = random.sample(support_meta, n_shot)
        n_shot_support = copy.deepcopy(n_shot_support_raw)
        operator = query['operator']
        operator_idx = operator_index[operator]
        for support in n_shot_support:    
            support['answer'] = support['answer'][operator_idx]
    elif dataset == 'open_mi':
        # use two classes for now
        query_class = query['answer']
        other_class = random.choice([cls for cls in query['classes'] if cls != query_class])
        order_keys = [query_class, other_class] if random.choice([True, False]) else [other_class, query_class]
        answers = {query_class: query_class, other_class: other_class}
        
        n_shot_support = []
        for i in range(n_shot):
            for key in order_keys:
                # For each key, add one shot
                support = {
                    'image': [query['support'][key]['images'][i]], 
                    'answer': answers[key],
                    'question': "This is a ",
                    'real_name': query['support'][key]['real_name'],
                    'mapping': f"{query['support'][key]['real_name']} maps to {answers[key]}"
                }
                n_shot_support.append(support)
    
    elif dataset == 'matching_mi':
        n_shot_support_raw = copy.deepcopy(random.sample(support_meta, n_shot))
        n_shot_support = []
        for i in range(n_shot):
            n_shot_support.append(n_shot_support_raw[i]['same'])
            n_shot_support.append(n_shot_support_raw[i]['diff'])
    else:
        n_shot_support = random.sample(support_meta, n_shot)
    return n_shot_support

def get_task_instruction(args):
    dataset = args.dataset
    description = args.task_description
    rule_instr = None
    if description == 'nothing':
        instr = ''
    
    if dataset == 'textocr':
        if description == 'concise':
            instr = 'Answer with the text inside the red box.'
        elif description == 'detailed':
            instr = 'An image will be provided where a red box is drawn around the text of interest. Answer with the text inside the red box. Ensure that the transcription is precise, reflecting the exact characters, including letters, numbers, symbols.'
    elif dataset == 'operator_induction':
        if description == 'concise':
            instr = 'Induce the mathematical operator and calculate the result.'
        elif description == 'detailed':
            instr = 'The image contains two digit numbers and a ? representing the mathematical operator. Induce the mathematical operator (addition, multiplication, minus) according to the results of the in-context examples and calculate the result.'
    elif dataset == 'operator_induction_interleaved':
        if description == 'concise':
            instr = 'Induce the mathematical operator between the two images and calculate the result.'
        elif description == 'detailed':
            instr = 'There are two input images, each representing a single digit number. Induce the mathematical operator (addition, multiplication, minus) that is applied between the two images according to the results of the in-context examples. Calculate the result for the new query images.'
    elif dataset == 'open_mi':
        if description == 'concise':
            instr = 'Answer the question with a single word or phase.'
        elif description == 'detailed':
            instr = "Induce the concept from the in-context examples. Answer the question with a single word or phase."
        elif description == 'rule':
            instr = "You will see images labeled with invented names from this set: 'blicket', 'dax', 'shously', 'perpo', or 'slation'. Each name corresponds to a specific object category. Learn which name matches which object type from the examples. Then, identify the object in the test image and respond with its corresponding invented name. Answer with only that single word."
        rule_instr = "You will be shown images of animals. Your task is to label each image using exactly one invented category name from the set [blicket, dax, shously, perpo, slation]. Each invented name maps to exactly one real-world animal category. Known mappings: {}. If you are uncertain, make your best guess using only the allowed labels. Respond with exactly one label from [blicket, dax, shously, perpo, slation]"
    elif dataset == 'clevr':
        if description == 'concise':
            instr = 'Find objects of the given type, induce what operation to use and calculate the result.'
        elif description == 'detailed':
            instr = 'The image contains objects of different shapes, colors, sizes and materials. The question describes the attribute and its value. You need to find all objects within the image that satisfy the condition. You should induce what operation to use according to the results of the in-context examples and then calculate the result.'
    elif dataset == 'matching_mi':
        if description == 'concise':
            instr = 'Determine the output for the new pair of images.'
        elif description == 'detailed':
            instr = 'According to the few-shot examples, induce what operation to do and determine the output for the two new images. Answer with "yes" or "no".'

    return rule_instr,instr

def format_answer(answer, dataset, query=None):
    if dataset in ['operator_induction', "clevr", 'operator_induction_interleaved']:
        answer = str(answer)
    return answer