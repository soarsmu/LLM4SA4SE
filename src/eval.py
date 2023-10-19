import pandas as pd
import os
import argparse
from sklearn.metrics import f1_score
import icecream as ic
import re

llama_pattern = r'(?:Label|Sentiment):\s*(\w+)'

def calculate_f1(test_df, cur_label_list):
    micro_f1 = f1_score(test_df['label'], test_df['pred'], labels=cur_label_list, average='micro')
    micro_f1 = round(micro_f1, 2)
    macro_f1 = f1_score(test_df['label'], test_df['pred'], labels=cur_label_list, average='macro')
    macro_f1 = round(macro_f1, 2)
    return macro_f1, micro_f1

def general_zero_shot():
    for index, row in test_df.iterrows():
        result_file = os.path.join(
            result_folder, '{}.txt'.format(index)
        )
        
        with open(result_file, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            found_sentiment = False
            multi_sent = False
            senti_count = dict()
            
            for token in last_line.split():
                token = token.lower()
                if 'positive' in token:
                    if senti_count.get('negative', 0) > 0:
                        multi_sent = True
                        break
                    test_df.loc[index, 'pred'] = 1
                    senti_count['positive'] = senti_count.get('positive', 0) + 1
                    found_sentiment = True
                elif 'negative' in token:
                    if senti_count.get('positive', 0) > 0:
                        multi_sent = True
                        break
                    test_df.loc[index, 'pred'] = 2
                    found_sentiment = True
                    senti_count['negative'] = senti_count.get('negative', 0) + 1
                elif 'neutral' in token:
                    if senti_count.get('positive', 0) > 0 or senti_count.get('negative', 0) > 0:
                        multi_sent = True
                        break
                    test_df.loc[index, 'pred'] = 0
                    found_sentiment = True
                    break
        if not found_sentiment or multi_sent:
            print('No sentiment found for {}'.format(index))
            test_df.loc[index, 'pred'] = 0
    
    cur_label_list = test_df['pred'].unique()
    test_df.to_csv(os.path.join(result_folder, 'prediction.csv'), index=False)
    # print(classification_report(test_df['label'], test_df['pred'], labels=cur_label_list))
    return calculate_f1(test_df, cur_label_list)
    

def jira_zero_shot():
    count_neutral = 0
    for index, row in test_df.iterrows():
        result_file = os.path.join(
            result_folder, '{}.txt'.format(index)
        )
        
        with open(result_file, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            found_sentiment = False
            multi_sent = False
            senti_count = dict()
            
            for token in last_line.split():
                token = token.lower()
                if 'positive' in token:
                    if senti_count.get('negative', 0) > 0:
                        multi_sent = True
                        break
                    test_df.loc[index, 'pred'] = 1
                    senti_count['positive'] = senti_count.get('positive', 0) + 1
                    found_sentiment = True
                elif 'negative' in token:
                    if senti_count.get('positive', 0) > 0:
                        multi_sent = True
                        break
                    test_df.loc[index, 'pred'] = 0
                    found_sentiment = True
                    senti_count['negative'] = senti_count.get('negative', 0) + 1
                elif 'neutral' in token:
                    count_neutral += 1
                    
        if not found_sentiment or multi_sent:
            print('No sentiment found for {}'.format(index))
            ground_truth = test_df.loc[index, 'label']
            test_df.loc[index, 'pred'] = 1 - ground_truth
    print('Neutral count:', count_neutral)    
    cur_label_list = test_df['pred'].unique()
    test_df.to_csv(os.path.join(result_folder, 'prediction.csv'), index=False)
    return calculate_f1(test_df, cur_label_list)

def code_zero_shot():
    for index, row in test_df.iterrows():
        result_file = os.path.join(
            result_folder, '{}.txt'.format(index)
        )
        
        with open(result_file, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            found_sentiment = False
            multi_sent = False
            senti_count = dict()
            
            for token in last_line.split():
                token = token.lower()
                if 'non-negative' in token:
                    test_df.loc[index, 'pred'] = 0
                    found_sentiment = True
                    break
                if 'positive' in token:
                    if senti_count.get('negative', 0) > 0:
                        multi_sent = True
                        break
                    test_df.loc[index, 'pred'] = 0
                    senti_count['positive'] = senti_count.get('positive', 0) + 1
                    found_sentiment = True
                elif 'negative' in token:
                    if senti_count.get('positive', 0) > 0:
                        multi_sent = True
                        break
                    test_df.loc[index, 'pred'] = 1
                    found_sentiment = True
                    senti_count['negative'] = senti_count.get('negative', 0) + 1
                elif 'neutral' in token:
                    if senti_count.get('positive', 0) > 0 or senti_count.get('negative', 0) > 0:
                        multi_sent = True
                        break
                    test_df.loc[index, 'pred'] = 0
                    found_sentiment = True
                    break
        if not found_sentiment or multi_sent:
            print('No sentiment found for {}'.format(index))
            test_df.loc[index, 'pred'] = 0
    
    cur_label_list = test_df['pred'].unique()
        
    test_df.to_csv(os.path.join(result_folder, 'prediction.csv'), index=False)
    return calculate_f1(test_df, cur_label_list)
    
    
def general_few_shot():    
    test_df = pd.read_csv(test_csv)
    for index, row in test_df.iterrows():
        result_file = os.path.join(
            result_folder, '{}.txt'.format(index)
        )
        found_sentiment = False
        with open(result_file, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            
            for token in last_line.split():
                token = token.lower()
                if 'positive' in token:                
                    test_df.loc[index, 'pred'] = 1                   
                    found_sentiment = True
                    break
                elif 'negative' in token:
                    test_df.loc[index, 'pred'] = 2
                    found_sentiment = True
                    break
                elif 'neutral' in token:                   
                    test_df.loc[index, 'pred'] = 0
                    found_sentiment = True
                    break
        if not found_sentiment:
            print('No sentiment found for {}'.format(index))
            test_df.loc[index, 'pred'] = 0
    
    cur_label_list = test_df['pred'].unique()
    test_df.to_csv(os.path.join(result_folder, 'prediction.csv'), index=False)
    return calculate_f1(test_df, cur_label_list)
    
def jira_few_shot():    
    test_df = pd.read_csv(test_csv)
    neutral_count = 0
    for index, row in test_df.iterrows():
        result_file = os.path.join(
            result_folder, '{}.txt'.format(index)
        )
        found_sentiment = False
        with open(result_file, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            
            for token in last_line.split():
                token = token.lower()
                if 'positive' in token:                
                    test_df.loc[index, 'pred'] = 1                   
                    found_sentiment = True
                    break
                elif 'negative' in token:
                    test_df.loc[index, 'pred'] = 0
                    found_sentiment = True
                    break
                elif 'neutral' in token:
                    ground_truth = test_df.loc[index, 'label']              
                    test_df.loc[index, 'pred'] = 1 - ground_truth
                    found_sentiment = True
                    neutral_count += 1
                    print('Neutral found for {}'.format(index))
                    break
                
        if not found_sentiment:
            print('No sentiment found for {}'.format(index))
            ground_truth = test_df.loc[index, 'label']
            test_df.loc[index, 'pred'] = 1 - ground_truth

    print('Neutral count:', neutral_count)
    cur_label_list = test_df['pred'].unique()
    test_df.to_csv(os.path.join(result_folder, 'prediction.csv'), index=False)
    return calculate_f1(test_df, cur_label_list)

def code_few_shot():    
    test_df = pd.read_csv(test_csv)
    for index, row in test_df.iterrows():
        result_file = os.path.join(
            result_folder, '{}.txt'.format(index)
        )
        found_sentiment = False
        with open(result_file, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            
            for token in last_line.split():
                token = token.lower()
                if 'non-negative' in token:
                    test_df.loc[index, 'pred'] = 0
                    found_sentiment = True
                    break
                elif 'positive' in token:
                    test_df.loc[index, 'pred'] = 0                   
                    found_sentiment = True
                    break
                elif 'negative' in token:
                    test_df.loc[index, 'pred'] = 1
                    found_sentiment = True
                    break
                elif 'neutral' in token:                   
                    test_df.loc[index, 'pred'] = 0
                    found_sentiment = True
                    break
        if not found_sentiment:
            print('No sentiment found for {}'.format(index))
            test_df.loc[index, 'pred'] = 0
    
    cur_label_list = test_df['pred'].unique()
    test_df.to_csv(os.path.join(result_folder, 'prediction.csv'), index=False)
    return calculate_f1(test_df, cur_label_list)
    
    
def general_llama():
    test_df = pd.read_csv(test_csv)
    for index, row in test_df.iterrows():
        result_file = os.path.join(
            result_folder, '{}.txt'.format(index)
        )
        
        found_sentiment = False
        with open(result_file, 'r') as f:
            lines = f.readlines()
            begin_line_index = -1            
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i]
                if '[/INST]' in line:
                    begin_line_index = i
                    break
            if begin_line_index == -1:
                print('Error happens {}'.format(index))
                test_df.loc[index, 'pred'] = 0
                continue
            
            for line in lines[begin_line_index:]:
                line = line.strip().lower()
                match = re.search(llama_pattern, line)
                if match:
                    result = match.group(1).strip()
                    if result == 'positive':
                        test_df.loc[index, 'pred'] = 1
                        found_sentiment = True
                        break
                    elif result == 'negative':
                        test_df.loc[index, 'pred'] = 2
                        found_sentiment = True
                        break
                    elif result == 'neutral':
                        test_df.loc[index, 'pred'] = 0
                        found_sentiment = True
                        break
                    
                for token in line.split():
                    if 'positive' in token:
                        test_df.loc[index, 'pred'] = 1
                        found_sentiment = True
                        break
                    elif 'negative' in token:
                        test_df.loc[index, 'pred'] = 2
                        found_sentiment = True
                        break
                    elif 'neutral' in token:
                        test_df.loc[index, 'pred'] = 0
                        found_sentiment = True
                        break
                    
        if not found_sentiment:
            print('No sentiment found for {}'.format(index))
            test_df.loc[index, 'pred'] = 0
    
    cur_label_list = test_df['pred'].unique()
    test_df.to_csv(os.path.join(result_folder, 'prediction.csv'), index=False)
    return calculate_f1(test_df, cur_label_list)
    
def jira_llama():
    neural_count = 0
    test_df = pd.read_csv(test_csv)
    for index, row in test_df.iterrows():
        result_file = os.path.join(
            result_folder, '{}.txt'.format(index)
        )
        
        found_sentiment = False
        with open(result_file, 'r') as f:
            lines = f.readlines()
            begin_line_index = -1            
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i]
                if '[/INST]' in line:
                    begin_line_index = i
                    break
            if begin_line_index == -1:
                print('Error happens {}'.format(index))
                test_df.loc[index, 'pred'] = 0
                continue
            
            for line in lines[begin_line_index:]:
                line = line.strip().lower()
                match = re.search(llama_pattern, line)
                if match:
                    result = match.group(1).strip()
                    if 'positive' in result:
                        test_df.loc[index, 'pred'] = 1
                        found_sentiment = True
                        break
                    elif 'negative' in result:
                        test_df.loc[index, 'pred'] = 0
                        found_sentiment = True
                        break
                    elif 'neutral' in result:
                        test_df.loc[index, 'pred'] = 1 - test_df.loc[index, 'label']
                        neural_count += 1
                        found_sentiment = True
                        break
                    
                for token in line.split():
                    if 'positive' in token:
                        test_df.loc[index, 'pred'] = 1
                        found_sentiment = True
                        break
                    elif 'negative' in token:
                        test_df.loc[index, 'pred'] = 0
                        found_sentiment = True
                        break
                    elif 'neutral' in token:
                        test_df.loc[index, 'pred'] = 1 - test_df.loc[index, 'label']
                        neural_count += 1
                        found_sentiment = True
                        break
                           
        if not found_sentiment:
            print('No sentiment found for {}'.format(index))
            ground_truth = test_df.loc[index, 'label']
            test_df.loc[index, 'pred'] = 1 - ground_truth
    print('Neutral count:', neural_count)
    cur_label_list = test_df['pred'].unique()
    test_df.to_csv(os.path.join(result_folder, 'prediction.csv'), index=False)
    return calculate_f1(test_df, cur_label_list)
    
def code_llama():
    test_df = pd.read_csv(test_csv)
    for index, row in test_df.iterrows():
        result_file = os.path.join(
            result_folder, '{}.txt'.format(index)
        )
        
        found_sentiment = False
        with open(result_file, 'r') as f:
            lines = f.readlines()
            begin_line_index = -1            
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i]
                if '[/INST]' in line:
                    begin_line_index = i
                    break
            if begin_line_index == -1:
                print('Error happens {}'.format(index))
                test_df.loc[index, 'pred'] = 0
                continue
            
            for line in lines[begin_line_index:]:
                line = line.strip().lower()
                match = re.search(llama_pattern, line)
                if match:
                    result = match.group(1).strip()
                    if result == 'non-negative':
                        test_df.loc[index, 'pred'] = 0
                        found_sentiment = True
                        break
                    if result == 'positive':
                        test_df.loc[index, 'pred'] = 0
                        found_sentiment = True
                        break
                    elif result == 'negative':
                        test_df.loc[index, 'pred'] = 1
                        found_sentiment = True
                        break
                    elif result == 'neutral':
                        test_df.loc[index, 'pred'] = 0
                        found_sentiment = True
                        break
                    
                for token in line.split():
                    if 'non-negative' in token:
                        test_df.loc[index, 'pred'] = 0
                        found_sentiment = True
                        break
                    if 'positive' in token:
                        test_df.loc[index, 'pred'] = 0
                        found_sentiment = True
                        break
                    elif 'negative' in token:
                        test_df.loc[index, 'pred'] = 1
                        found_sentiment = True
                        break
                    elif 'neutral' in token:
                        test_df.loc[index, 'pred'] = 0
                        found_sentiment = True
                        break
                    
        if not found_sentiment:
            print('No sentiment found for {}'.format(index))
            test_df.loc[index, 'pred'] = 0
    
    cur_label_list = test_df['pred'].unique()
    test_df.to_csv(os.path.join(result_folder, 'prediction.csv'), index=False)
    return calculate_f1(test_df, cur_label_list)
    
def check_few_shot_result():
    test_df = pd.read_csv(test_csv)
    for index, row in test_df.iterrows():
        result_file = os.path.join(
            result_folder, '{}.txt'.format(index)
        )        
        with open(result_file, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]            
            if not 'ASSISTANT:' in last_line:
                print('Error happens {}'.format(index))
                
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # args.add_argument('--dataset', '-d', type=str, required=True)
    # args.add_argument('--model', '-m', type=str, required=True)
    # args.add_argument('--prompt', '-p', type=str, required=True)
    args.add_argument('--shot', '-s', type=int, required=True)
    args = args.parse_args()
    # prompt_variant = args.prompt
    # model_name = args.model
    shot_type = args.shot
    # dataset = args.dataset
    # model_list = ['vicuna', 'wizardlm', 'llama2']
    model_list = ['vicuna']
    # dataset_list = ['code', 'github', 'app', 'jira', 'so']
    dataset_list = ['jira']
    
    if shot_type < 1:
        prompt_list = [7, 1, 2]
    else:
        prompt_list = [1, 3, 5]
    
    for model_name in model_list:
        for dataset in dataset_list:
            macro_f1_list = []
            micro_f1_list = []
            
            for prompt_variant in prompt_list:
                if shot_type < 1:
                    shot_variant = prompt_variant          
                    if not dataset == 'jira':
                        prompt_variant = model_name + '-' + str(prompt_variant)
                    else:
                        prompt_variant = model_name + '-jira-' + str(prompt_variant)
                
                    if model_name == 'wizardlm':
                        prompt_variant = 'vicuna-' + '-'.join(prompt_variant.split('-')[1:])
                else:
                    shot_variant = prompt_variant
                    prompt_variant = model_name
                    if model_name == 'wizardlm':
                        prompt_variant = 'vicuna'
                    if dataset == 'jira':
                        prompt_variant = prompt_variant + '-jira'
                        
                print('Extracting sentiment for {} {} {} {}'.format(dataset, model_name, prompt_variant, shot_variant))
    
                if dataset == 'app':
                    test_csv = '../data/{}-test.csv'.format(dataset)
                else:
                    test_csv = '../data/{}-sampled-test.csv'.format(dataset)
                    
                test_df = pd.read_csv(test_csv)
                
                if shot_type < 1:
                    result_folder = '../results/{}/{}/zero-shot/{}/'.format(dataset, model_name, prompt_variant)
                    if model_name == 'llama2':
                        if dataset == 'jira':
                            cur_macro, cur_micro = jira_llama()
                        elif dataset == 'code':
                            cur_macro, cur_micro = code_llama()
                        else:
                            cur_macro, cur_micro = general_llama()
                    else:
                        if dataset == 'jira':
                            cur_macro, cur_micro = jira_zero_shot()
                        elif dataset == 'code':
                            cur_macro, cur_micro = code_zero_shot()
                        else:
                            cur_macro, cur_micro = general_zero_shot()
                    macro_f1_list.append(cur_macro)
                    micro_f1_list.append(cur_micro)                    
                else:
                    result_folder = '../results/{}/{}/few-shot/{}/{}'.format(dataset, model_name, prompt_variant, shot_variant)
                    if model_name == 'llama2':
                        if dataset == 'jira':
                            cur_macro, cur_micro = jira_llama()
                        elif dataset == 'code':
                            cur_macro, cur_micro = code_llama()
                        else:
                            cur_macro, cur_micro = general_llama()
                    else:
                        if dataset == 'jira':
                            cur_macro, cur_micro = jira_few_shot()
                        elif dataset == 'code':
                            cur_macro, cur_micro = code_few_shot()
                        else:
                            cur_macro, cur_micro = general_few_shot()
                    macro_f1_list.append(cur_macro)
                    micro_f1_list.append(cur_micro)
            print('------------------' * 3)
            print(model_name, dataset)
            print('Macro-F1 List:')
            for item in macro_f1_list:
                print(item)
            print('Micro-F1 List:')
            for item in micro_f1_list:
                print(item)
            print('==================' * 3)