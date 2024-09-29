import json
import argparse
import os
from tqdm import tqdm

def main(args):
    model_name = args.model_path[args.model_path.rfind('/') + 1:] if args.model_path.find('/') != -1 else args.model_path
    with open(f"../infer_data/{model_name}/{args.method}/{args.dataset}.json", "r", encoding='utf-8') as fp:
        datasets = json.load(fp)
    total_len = len(datasets)
    right = 0
    for data in tqdm(datasets):
        solution = data["solution"]
        solution = solution[solution.find("```python") + len("```python"):]
        solution = solution[:solution.find("```")]
        for key, value in data["variables"].items():
            if isinstance(value, str):
                solution = f'{key} = {repr(value)}\n' + solution
            else:
                solution = f'{key} = {value}\n'  + solution
        answer = {}
        try:
            exec(solution, answer)
        except:
            total_len -= 1
            continue
        
        if str(answer["answer"]) == data["response"]:
            right += 1
        elif str(answer["answer"]) == 'True' and ('yes' == data["response"].lower() or 'true' == data["response"].lower()):
            right += 1
        elif str(answer["answer"]) == 'False' and ('no' == data["response"].lower() or 'false' == data["response"].lower()):
            right += 1
    print(f"acc = {right / total_len}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help='dataset name')
    parser.add_argument('--model_path', required=True, type=str, help='model path')
    parser.add_argument('--method', type=str, default='raw',help='prompt engineering technique')
    args = parser.parse_args()

    main(args)