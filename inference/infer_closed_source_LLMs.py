import json
from transformers import pipeline
import torch
import argparse
from licloud.apihub import Model
from licloud.langchain.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import licloud
from tqdm import tqdm
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
    os.makedirs(f"../infer_data/{args.model_path}/{args.method}", exist_ok=True)
    try:
        with open(f'../infer_data/{args.model_path}/{args.method}/{args.dataset}.json', 'r') as fp:
            infer_data = json.load(fp)
    except:
        infer_data = []
    
    if args.model_path == 'GPT_35':
        LLM = Model.GPT_35
    if args.model_path == 'GPT_4O':
        LLM = Model.GPT_4O
    if args.model_path == 'GPT_4':
        LLM = Model.GPT_4
    if args.model_path == 'GPT_4_TURBO':
        LLM = Model.GPT_4_TURBO
    if args.model_path == 'CLAUDE_3_SONNET':
        LLM = Model.CLAUDE_3_SONNET
    if args.model_path == 'CLAUDE_2':
        LLM = Model.CLAUDE_2

    data_length = len(infer_data)//3
    if datasets[data_length : ] == []:
        raise ValueError("You've finished generating!")

    for data in tqdm(datasets[data_length : ]):
        for query in data["query"]:
            formatted_query = custom_format(query, data["variables"])
            if args.method == 'raw':
                prompt = f"Solve the question below.\n\n{formatted_query}\n\nProvide the final result like this:\n```llm_result\nxxx\n```\n"

            elif args.method == 'pot':
                prompt = f"Write a Python code to solve the question below.\n\n{formatted_query}\n\nJust provide the code, no explainations."

            elif args.method == 'cot':
                prompt = f"Solve the question below.\n\n{formatted_query}\n\nUse chain-of-thought to solve it, and make sure to provide the final result like this:\n```llm_result\nxxx\n```\n"

            else:
                raise ValueError("method not found")
            
            message = [
                HumanMessage(content=prompt),
            ]
            chat = AzureChatOpenAI(model=LLM)
            try:
                each_result = chat.invoke(message)
                each_result = each_result.content
            except:
                try:
                    each_result = chat.invoke(message)
                    each_result = each_result.content
                except:
                    each_result = ''
            if args.method == 'cot' or args.method == 'raw':
                each_result = each_result[each_result.find('```llm_result') + len('```llm_result'):]
                each_result = each_result[:each_result.find('```')]
                each_result = each_result.strip("\n")
            elif args.method == 'pot':
                if "```python" in each_result:
                    each_result = each_result[each_result.find('```python') + len('```python'):]
                    each_result = each_result[:each_result.find('```')]
                each_result = each_result.strip()
            infer_data.append({"prompt": prompt, "solution": data["solution"], "response": each_result, "variables": data["variables"]})

    
        with open(f'../infer_data/{args.model_path}/{args.method}/{args.dataset}.json', 'w') as fp:
            json.dump(infer_data, fp, ensure_ascii=False, indent = 4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help='dataset name')
    parser.add_argument('--temperature', default=0.8, type=float, help='inference temperature')
    parser.add_argument('--top_p', default=0.95, type=float, help='inference top_p')
    parser.add_argument('--model_path', required=True, type=str, help='model path')
    parser.add_argument('--tensor_parallel_size', default=1, type=int, help='gpu numbers')
    parser.add_argument('--method', default='raw', type=str, help='prompt engineering technique')
    args = parser.parse_args()
    while(1):
        try:
            main(args)
        except licloud.apihub._api_hub.APIHubException:
            continue
        except json.decoder.JSONDecodeError:
            continue
        except:
            break
