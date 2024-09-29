import json
from vllm import LLM, SamplingParams
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
    with open(f'../data/test/{args.dataset}.json', 'r') as fp:
        datasets = json.load(fp)
    model_name = args.model_path[args.model_path.rfind('/') + 1:] if args.model_path.find('/') != -1 else args.model_path
    os.makedirs(f"../infer_data/{model_name}/{args.method}", exist_ok=True)
    infer_data = []
    prompts_all = []
    for data in datasets:
        for query in data["query"]:
            formatted_query = custom_format(query, data["variables"])
            if args.method == 'raw':
                prompt = f"{formatted_query}\n\nProvide the final result like this:\n```llm_result\nxxx\n```\n"
                max_tokens = 512
            elif args.method == 'pot':
                prompt = f"Write Python code to solve the question below:\n\n[Question] {formatted_query}."
                max_tokens = 512
            elif args.method == 'cot':
                prompt = f"Use Chain-of-Thought reasoning, solve the question below step-by-step:\n\n[Question] {formatted_query}\n\nProvide the final result like this:\n```llm_result\nxxx\n```\n"
                max_tokens = 1024
            else:
                raise ValueError("method not found")
            infer_data.append({"prompt": prompt, "solution": data["solution"], "variables": data["variables"]})
            prompts_all.append(prompt)

    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, max_model_len=2048, enforce_eager=True)
    
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=max_tokens)
    responses = llm.generate(prompts_all, sampling_params=sampling_params)
    
    for i in range(len(responses)):
        each_result = responses[i].outputs[0].text
        # infer_data[i]["response"] = each_result
        if args.method == 'cot' or args.method == 'raw':
            if each_result.find('```llm_result\n') != -1:
                each_result = each_result[each_result.find('```llm_result\n') + len('```llm_result\n'):]
                each_result = each_result[:each_result.find('```')]
            elif each_result.find('```llm_result ') != -1:
                each_result = each_result[each_result.find('```llm_result ') + len('```llm_result '):]
                each_result = each_result[:each_result.find('```')]
            elif each_result.find("```\nllm_result\n") != -1:
                each_result = each_result[each_result.find("```\nllm_result\n") + len("```\nllm_result\n"):]
                each_result = each_result[:each_result.find('```')]
            elif each_result.find("```\nllm_result ") != -1:
                each_result = each_result[each_result.find("```\nllm_result ") + len("```\nllm_result "):]
                each_result = each_result[:each_result.find('```')]
            elif each_result.find("```Llm_result\n") != -1:
                each_result = each_result[each_result.find("```Llm_result\n") + len("```Llm_result\n"):]
                each_result = each_result[:each_result.find('```')]  
            elif each_result.find("```Llm_result ") != -1:
                each_result = each_result[each_result.find("```Llm_result ") + len("```Llm_result "):]
                each_result = each_result[:each_result.find('```')]
            elif each_result.find("The final answer is:\n") != -1:
                each_result = each_result[each_result.find("The final answer is:\n") + len("The final answer is:\n") : ]
                each_result = each_result.strip("\n")
                each_result = each_result[:each_result.find("\n")]
            elif each_result.find("The final answer is ") != -1:
                each_result = each_result[each_result.find("The final answer is ") + len("The final answer is "):]
                each_result = each_result[:each_result.find('.')]
            else:
                each_result = each_result[each_result.find('```llm_result\n') + len('```llm_result\n'):]
                each_result = each_result[:each_result.find('```')]
            each_result = each_result.strip("\n")
        elif args.method == 'pot':
            if "```python" in each_result:
                each_result = each_result[each_result.find('```python') + len('```python'):]
            else:
                each_result = each_result[each_result.find('```') + len('```'):]
            each_result = each_result[:each_result.find('```')]
            each_result = each_result.strip()
        
        infer_data[i]["response"] = each_result
    
    with open(f"../infer_data/{model_name}/{args.method}/{args.dataset}.json", 'w',encoding='utf-8') as fp:
        json.dump(infer_data, fp, ensure_ascii=False, indent = 4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help='dataset name')
    parser.add_argument('--model_path', required=True, type=str, help='model path')
    parser.add_argument('--temperature', default=0.8, type=float, help='inference temperature')
    parser.add_argument('--top_p', default=0.95, type=float, help='inference top_p')
    parser.add_argument('--tensor_parallel_size', default=1, type=int, help='gpu numbers')
    parser.add_argument('--method', default='raw', type=str, help='prompt engineering technique')
    args = parser.parse_args()

    main(args)