import json
import os
import sys

# Assume 'evals' directory is in the same parent directory as this script,
# or add the path to sys.path if it's elsewhere.
# For example, if your eval.py is in /path/to/VL-ICL/evals, and this script
# is in /path/to/your_script.py, you might need:
# sys.path.append(os.path.join(os.path.dirname(__file__), 'VL-ICL'))
# Make sure the 'evals' module is discoverable.
try:
    from evals import eval
except ImportError:
    print("Error: Could not import 'evals.eval'.")
    print("Please ensure 'evals' directory is in your Python path or in the same directory.")
    print("You might need to adjust sys.path or the script's location.")
    sys.exit(1)


# Define the metrics dictionary as in your original script
metrics = {
    "textocr": "Acc",
    "operator_induction": "Acc",
    "open_mi": "Acc",
    "clevr": "Acc",
    'operator_induction_interleaved': "Acc",
    'matching_mi': "Acc",
}

def collect_and_evaluate_results(base_results_dir, datasets_directory, model, datasets, shots, configurations):
    """
    Collects, evaluates, and compiles results for specified models.

    Args:
        base_results_dir (str): The base directory where 'w_img' and 'wo_img' folders are located.

    Returns:
        dict: A dictionary containing compiled results.
              Format: {
                  "model_name": {
                      "dataset_name": {
                          "shot_value": {
                              "w_img": score,
                              "wo_img": score
                          }
                      }
                  }
              }
    """

    compiled_results = {}
    output_dict = None
    # output_dicts[model] = {}
    for config in configurations:
        print(f"Processing image status folder: {config}")
        current_img_dir = os.path.join(base_results_dir, config)
        if config == 'perception_contain' or config == 'perception_llm_judge':
            current_img_dir = os.path.join(base_results_dir, 'perception')
        if not os.path.isdir(current_img_dir):
            print(f"Warning: Directory not found: {current_img_dir}. Skipping.")
            continue

        for dataset in datasets:
            # output_dicts[model][dataset] = {}
            print(f"Processing dataset: {dataset}")
            dataset_dir = os.path.join(current_img_dir, dataset)
            if not os.path.isdir(dataset_dir):
                print(f"Warning: Directory not found: {dataset_dir}. Skipping.")
                continue

            print(f"Processing model: {model}")
            dataset_dir = os.path.join (dataset_dir,model)
            if model not in compiled_results:
                compiled_results[model] = {}
            if dataset not in compiled_results[model]:
                compiled_results[model][dataset] = {}

            if config == 'perception_llm_judge':
                meta_data_dir = os.path.join(datasets_directory, dataset,'meta.json')
                with open(meta_data_dir, "r") as f:
                    meta_data = json.load(f)
                result_filepath = os.path.join(dataset_dir, "perception.json")
                with open(result_filepath, "r") as f:
                    results_dict = json.load(f)
                score, output_dict = eval.eval_perception_llm_judge(results_dict, dataset, meta_data)
                print(f"Processed: {config}/{dataset}/perception.json -> Score: {score * 100.0:.2f} {metrics[dataset]}")
                compiled_results[model][dataset][config] = score
                continue
            elif config == 'perception_contain':
                meta_data_dir = os.path.join(datasets_directory, dataset,'meta.json')
                with open(meta_data_dir, "r") as f:
                    meta_data = json.load(f)
                result_filepath = os.path.join(dataset_dir, "perception.json")
                with open(result_filepath, "r") as f:
                    results_dict = json.load(f)
                score, output_dict = eval.eval_perception_contain(results_dict, dataset, meta_data)
                print(f"Processed: {config}/{dataset}/perception.json -> Score: {score * 100.0:.2f} {metrics[dataset]}")
                compiled_results[model][dataset][config] = score
                continue
            elif config == 'perception_classification':
                result_filepath = os.path.join(dataset_dir, "perception_classification.json")
                with open(result_filepath, "r") as f:
                    results_dict = json.load(f)
                score = eval.eval_perception_classification(results_dict, dataset)
                print(f"Processed: {config}/{dataset}/perception_classification.json -> {score}")
                
                compiled_results[model][dataset][config] = score
                continue
            else:
                if config == "rule_only_query_desc" or config == "rule_only":
                    compiled_results[model][dataset][config] = {}
                    result_filename = f"1-shot.json"
                    result_filepath = os.path.join(dataset_dir, result_filename)
                    if os.path.exists(result_filepath):
                        try:
                            with open(result_filepath, "r") as f:
                                results_dict = json.load(f)
                            
                            score = eval.eval_scores(results_dict, dataset)
                            print(f"Processed: {config}/{dataset}/{result_filename} -> Score: {score * 100.0:.2f} {metrics[dataset]}")
                            
                            # Store the score
                            compiled_results[model][dataset][config] = round(score * 100.0, 2)
                        except json.JSONDecodeError:
                            print(f"Error: Could not decode JSON from {result_filepath}. Skipping.")
                        except Exception as e:
                            print(f"An error occurred while processing {result_filepath}: {e}. Skipping.")
                    else:
                        print(f"File not found: {result_filepath}. Skipping.")
                        pass
                else:
                    for shot in shots:
                        print(f"Processing shot: {shot}")
                        if str(shot) not in compiled_results[model][dataset]:
                            compiled_results[model][dataset][str(shot)] = {}
                        result_filename = f"{shot}-shot.json"
                        result_filepath = os.path.join(dataset_dir, result_filename)

                        if os.path.exists(result_filepath):
                            try:
                                with open(result_filepath, "r") as f:
                                    results_dict = json.load(f)
                                
                                score = eval.eval_scores(results_dict, dataset)
                                print(f"Processed: {config}/{dataset}/{result_filename} -> Score: {score * 100.0:.2f} {metrics[dataset]}")
                                
                                # Store the score
                                compiled_results[model][dataset][str(shot)][config] = round(score * 100.0, 2)
                            except json.JSONDecodeError:
                                print(f"Error: Could not decode JSON from {result_filepath}. Skipping.")
                            except Exception as e:
                                print(f"An error occurred while processing {result_filepath}: {e}. Skipping.")
                        else:
                            print(f"File not found: {result_filepath}. Skipping.")
                            pass # Silently skip missing files as per your requirement
                            
    return compiled_results, output_dict

if __name__ == "__main__":

    base_results_directory = "/home/eml/yiran.huang/VL-ICL/results"
    datasets_directory = "/lustre/groups/eml/datasets/VL-ICL/"
    target_models = ['qwen2.5-vl-3b','qwen2.5-vl-7b']
    shots = [0, 1, 2, 4, 5]
    datasets = ['open_mi']
    configurations = ['w_img', 'w_blank_query_img', 'w_blank_demo_img', 'w_blank_img_all', 'rule_only_query_desc','rule_only','rule_demo','demo_desc_query_img', 'demo_desc_query_desc', 'demo_img_desc_query_img', 'demo_img_desc_query_img_desc', 'demo_img_desc_query_img_desc_after_labels','perception_classification','perception_contain']
    # configurations = ["demo_desc_query_desc"]

    # configurations = ['perception_llm_judge']
    for model in target_models:
        compiled_data, output_dict = collect_and_evaluate_results(base_results_directory, datasets_directory, model, datasets, shots, configurations)
        output_json_dir = f"{os.path.dirname(os.path.abspath(__file__))}/results/{model}"
        if not os.path.exists(output_json_dir):
            os.makedirs(output_json_dir)
        output_json_filename = f"compiled_evaluation_results_{model}_test.json"
        output_filepath = os.path.join(output_json_dir, output_json_filename)
        
        with open(output_filepath, "w") as f:
            json.dump(compiled_data, f, indent=4)
        print(f"\nEvaluation complete. All existing results compiled into: {output_filepath}")
        # if output_dict is not None:
        #     output_dict_filename = f"{model}_open_mi_perception_results_.json"
        #     output_dict_filepath = os.path.join(output_json_dir, output_dict_filename)
        #     with open(output_dict_filepath, "w") as f:
        #         json.dump(output_dict, f, indent=4)
        #     print(f"\nEvaluation complete. Perception results compiled into: {output_dict_filepath}")