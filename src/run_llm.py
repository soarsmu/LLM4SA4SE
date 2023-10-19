import pandas as pd
import icecream as ic
from tqdm import tqdm
import os
import time
from transformers import AutoTokenizer
import transformers
import argparse
import torch
import json
import logging


model_dict = {
    'vicuna': 'lmsys/vicuna-13b-v1.5',
    'wizardlm': 'WizardLM/WizardLM-13B-V1.2',
    'llama2': "meta-llama/Llama-2-13b-chat-hf"
}

domain_dict = {
    'github': 'GitHub',
    'so': 'Stack Overflow',
    'app': "APP reviews",
    'jira': "Jira",
    'code': "code reviews"
}

prompt_folder = '../prompt/'
result_folder = '../results/'


def generate_zero_shot(prompt_template, prompt_variant, shot_type):
    if dataset == 'app':
        test_csv = '../data/{}-test.csv'.format(dataset)
    else:
        test_csv = '../data/{}-sampled-test.csv'.format(dataset)
        
    test_df = pd.read_csv(test_csv)
        
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        if os.path.exists(os.path.join(result_folder, '{}.txt'.format(i))):
            continue
        
        prompt = prompt_template.format(domain_dict[dataset], row['sentence'])
        logging.info('The template for sentence is: {}'.format(prompt))
        
        tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name])
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_dict[model_name],
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        start = time.time()
        sequences = pipeline(
            prompt,
            max_length=2048,
            # do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
        
        logging.info('time taken: {}'.format(time.time() - start))
        result_file = os.path.join(result_folder, '{}.txt'.format(i))    
        with open(result_file, 'w') as f:
            for seq in sequences:
                # logging.info(seq['generated_text'])
                f.write(seq['generated_text'] + '\n')


def generate_few_shot(prompt_template, shot_type):
    if dataset == 'app':
        test_csv = '../data/{}-test.csv'.format(dataset)
    else:
        test_csv = '../data/{}-sampled-test.csv'.format(dataset)
    test_df = pd.read_csv(test_csv)
    
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        if os.path.exists(os.path.join(result_folder, '{}.txt'.format(i))):
            continue
        
        with open('../sampled-shot/{}-shot/{}/{}.txt'.format(shot_type, dataset, i), 'r') as f:
            demo = f.read()
        prompt_list = []
        prompt_list.append(domain_dict[dataset])
        prompt_list.append(demo)
        prompt_list.append(row['sentence'])
        prompt = prompt_template.format(*prompt_list)
        
        tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name])
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_dict[model_name],
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        
        start = time.time()
        sequences = pipeline(
            prompt,
            max_length=2048,
            # do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
        
        logging.info('time taken: {}'.format(time.time() - start))
        result_file = os.path.join(result_folder, '{}.txt'.format(i))    
        with open(result_file, 'w') as f:
            for seq in sequences:
                # logging.info(seq['generated_text'])
                f.write(seq['generated_text'] + '\n')
                
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', '-d', type=str, required=True)
    args.add_argument('--model', '-m', type=str, required=True)
    args.add_argument('--prompt', '-p', type=str, required=True)
    args.add_argument('--shot', '-s', type=int, required=True)
    
    args = args.parse_args()
    shot_type = args.shot
    dataset = args.dataset
    model_name = args.model
    prompt_variant = args.prompt
    
    if shot_type < 1:
        with open('../prompt/zero_shot_prompt.json', 'r') as f:
            prompt_tmps = json.load(f)
        result_folder = '../results/{}/{}/zero-shot/{}/'.format(dataset, model_name, prompt_variant)
    else:
        with open('../prompt/few_shot_prompt.json', 'r') as f:
            prompt_tmps = json.load(f)
        result_folder = '../results/{}/{}/few-shot/{}/{}'.format(dataset, model_name, prompt_variant, shot_type)
    os.makedirs(result_folder, exist_ok=True)
    prompt_template = prompt_tmps[prompt_variant]
    os.makedirs('../log', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s %(message)s', 
        handlers=[
            logging.FileHandler("../log/{}_{}_{}_{}_{}.log".format(dataset, model_name, prompt_variant, shot_type, time.time())),
            logging.StreamHandler()
        ]
    )
    if shot_type < 1:
        generate_zero_shot(prompt_template, prompt_variant, shot_type)
    else:
        generate_few_shot(prompt_template, shot_type)