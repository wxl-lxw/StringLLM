import json
import argparse
import os

def custom_format(template, variables):
    for key, value in variables.items():
        if isinstance(value, str):
            value = '`'+ value + '`'
            exec(f'{key} = {repr(value)}')
        else:
            exec(f'{key} = {value}')
    
    result = eval(f'f"""{template}"""')
    return result

def main(args):
    with open(f'{args.dataset}.json', 'r') as fp:
        datasets = json.load(fp)
    constructed_data = []
    for data in datasets:
        for query in data["query"]:
            each_data_point = {}
            formatted_query = custom_format(query, data["variables"])
            each_data_point["instruction"] = f"Write a Python code to solve the question below.\n\n{formatted_query}\n\nJust provide the code, no explainations."

            each_data_point["input"] = ""

            output_temp = data["solution"][len("```python\n"):]
            for key, value in data["variables"].items():
                if isinstance(value, str):
                    output_temp = f"{key} = {repr(value)}\n{output_temp}"
                else:
                    output_temp = f"{key} = {value}\n{output_temp}"
            each_data_point["output"] = f"```python\n{output_temp}"
            constructed_data.append(each_data_point)
    
    with open(f"{args.dataset}_alpaca.json", "w") as fp:
        json.dump(constructed_data, fp, ensure_ascii=False, indent = 4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help='dataset')
    args = parser.parse_args()

    main(args)