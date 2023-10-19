import pandas as pd
import os
import argparse
from sklearn.metrics import classification_report
import icecream as ic
import random

prompt_folder = '../prompt/'

general_label_list = [0, 1, 2]
code_jira_label_list = [0, 1]
model_list = ['vicuna', 'wizardlm', 'llama2']
dataset_list = ['code', 'github', 'app', 'jira', 'so']

model_template_dict = {
    'vicuna': 'vicuna',
    'wizardlm': 'vicuca',
    'llama2': 'llama2'
}

def analyze_hits():
    if shot_type < 1:
        prompt_variant = 7
    else:
        prompt_variant = 5
    test_size = {}
    vicuna_hits, vicuna_misses = set(), set()
    wizardlm_hits, wizardlm_misses = set(), set()
    llama2_hits, llama2_misses = set(), set()
    
    for dataset in dataset_list:        
        if shot_type < 1:
            result_folder = os.path.join('../results', dataset, 'vicuna/zero-shot/vicuna-7')
            if dataset == 'jira':
                result_folder = os.path.join('../results', dataset, 'vicuna/zero-shot/vicuna-jira-7')
        else:
            result_folder = os.path.join('../results', dataset, 'vicuna/few-shot/vicuna/5')
            if dataset == 'jira':
                result_folder = os.path.join('../results', dataset, 'vicuna/few-shot/vicuna-jira/5')
                
        test_csv = os.path.join(result_folder, 'prediction.csv')
        test_df = pd.read_csv(test_csv)
        if not dataset in test_size:
            test_size[dataset] = len(test_df)
        for i, row in test_df.iterrows():
            if row['label'] == row['pred']:
                vicuna_hits.add(dataset + '-' + str(i))
            else:
                vicuna_misses.add(dataset + '-' + str(i))
                
        if shot_type < 1:
            result_folder = os.path.join('../results', dataset, 'wizardlm/zero-shot/vicuna-7')
            if dataset == 'jira':
                result_folder = os.path.join('../results', dataset, 'wizardlm/zero-shot/vicuna-jira-7')
        else:
            result_folder = os.path.join('../results', dataset, 'wizardlm/few-shot/vicuna/5')
            if dataset == 'jira':
                result_folder = os.path.join('../results', dataset, 'wizardlm/few-shot/vicuna-jira/5')
            
        test_csv = os.path.join(result_folder, 'prediction.csv')
        test_df = pd.read_csv(test_csv)        
        for i, row in test_df.iterrows():
            if row['label'] == row['pred']:
                wizardlm_hits.add(dataset + '-' + str(i))
            else:
                wizardlm_misses.add(dataset + '-' + str(i))
                
        if shot_type < 1:
            result_folder = os.path.join('../results', dataset, 'llama2/zero-shot/llama2-7')
            if dataset == 'jira':
                result_folder = os.path.join('../results', dataset, 'llama2/zero-shot/llama2-jira-7')
        else:
            result_folder = os.path.join('../results', dataset, 'llama2/few-shot/llama2/5')
            if dataset == 'jira':
                result_folder = os.path.join('../results', dataset, 'llama2/few-shot/llama2-jira/5')
            
        test_csv = os.path.join(result_folder, 'prediction.csv')
        test_df = pd.read_csv(test_csv)        
        for i, row in test_df.iterrows():
            if row['label'] == row['pred']:
                llama2_hits.add(dataset + '-' + str(i))
            else:
                llama2_misses.add(dataset + '-' + str(i))
                
    # print('vicuna hits:', len(vicuna_hits))
    # print('wizardlm hits:', len(wizardlm_hits))
    # print('llama2 hits:', len(llama2_hits))
    
    # if len(vicuna_hits) + len(vicuna_misses) != len(wizardlm_hits) + len(wizardlm_misses):
    #     print('ERROR')
    # if len(vicuna_hits) + len(vicuna_misses) != len(llama2_hits) + len(llama2_misses):
    #     print('ERROR')
    # all_intersection = vicuna_hits.intersection(wizardlm_hits).intersection(llama2_hits)
    # print('all intersection:', len(all_intersection))
    # all_hits = vicuna_hits.union(wizardlm_hits).union(llama2_hits)
    # print('all hits:', len(all_hits))
    
    # vicuna_wizardlm_intersection = vicuna_hits.intersection(wizardlm_hits)
    # # print('vicuna wizardlm intersection percentage:', round(len(vicuna_wizardlm_intersection) / len(vicuna_hits.union(wizardlm_hits)), 3))
    
    # print('vicuna wizardlm intersection:', len(vicuna_wizardlm_intersection) - len(all_intersection))
    
    # vicuna_llama2_intersection = vicuna_hits.intersection(llama2_hits)
    # # print('vicuna llama2 intersection percentage:', round(len(vicuna_llama2_intersection) / len(vicuna_hits.union(llama2_hits)), 3))
    
    # print('vicuna llama2 intersection:', len(vicuna_llama2_intersection) - len(all_intersection))
    # wizardlm_llama2_intersection = wizardlm_hits.intersection(llama2_hits)
    
    # print('wizardlm llama2 intersection:', len(wizardlm_llama2_intersection) - len(all_intersection))
    # # print('wizardlm llama2 intersection percentage:', round(len(wizardlm_llama2_intersection) / len(llama2_hits.union(wizardlm_hits)), 3))
    
    # only_vicuna = vicuna_hits.difference(wizardlm_hits).difference(llama2_hits)
    # print('only vicuna:', len(only_vicuna))
    # only_wizardlm = wizardlm_hits.difference(vicuna_hits).difference(llama2_hits)
    # print('only wizardlm:', len(only_wizardlm))
    # only_llama2 = llama2_hits.difference(vicuna_hits).difference(wizardlm_hits)
    # print('only llama2:', len(only_llama2))
    
    ## failure analysis
    print('vicuna misses:', len(vicuna_misses))
    print('wizardlm misses:', len(wizardlm_misses))    
    print('llama2 misses:', len(llama2_misses))
    all_misses = vicuna_misses.union(wizardlm_misses).union(llama2_misses)
    common_misses = vicuna_misses.intersection(wizardlm_misses).intersection(llama2_misses)
    # print('all misses:', len(all_misses))
    print('common misses:', len(common_misses))
    common_misses = sorted(list(common_misses))
    dataset_wrong_dict = {}
    for miss in common_misses:
        dataset = miss.split('-')[0]
        if dataset not in dataset_wrong_dict:
            dataset_wrong_dict[dataset] = 0
        dataset_wrong_dict[dataset] += 1
    print(dataset_wrong_dict)
    # for dataset in dataset_wrong_dict.keys():
    #     print(dataset, dataset_wrong_dict[dataset], '(' + str(round(dataset_wrong_dict[dataset] / test_size[dataset] * 100, 1)) + '%)')
    with open('common_misses_{}.txt'.format(shot_type), 'w') as f:
        for miss in common_misses:
            f.write(miss + '\n')
                
def analyze_few_zero_misses():
    with open('common_misses_1.txt', 'r') as f:
        lines = f.readlines()
    few_miss_set = set()
    for line in lines:
        few_miss_set.add(line.strip())
    with open('common_misses_0.txt', 'r') as f:
        lines = f.readlines()
    zero_miss_set = set()
    for line in lines:
        zero_miss_set.add(line.strip())
    common_misses = few_miss_set.intersection(zero_miss_set)
    print('common misses:', len(common_misses))
    data_dict = {}
    for miss in common_misses:
        dataset = miss.split('-')[0]
        if dataset not in data_dict:
            data_dict[dataset] = 0
        data_dict[dataset] += 1
    print(data_dict)
        
    # few_unique_misses = few_miss_set.difference(zero_miss_set)
    # print('few unique misses:', len(few_unique_misses))
    # zero_unique_misses = zero_miss_set.difference(few_miss_set)
    # print('zero unique misses:', len(zero_unique_misses))
    # with open('few_unique_misses.txt', 'w') as f:
    #     for miss in sorted(list(few_unique_misses)):
    #         f.write(miss + '\n')
    # with open('zero_unique_misses.txt', 'w') as f:
    #     for miss in sorted(list(zero_unique_misses)):
    #         f.write(miss + '\n')
    # with open('common_misses.txt', 'w') as f:
    #     for miss in sorted(list(common_misses)):
    #         f.write(miss + '\n')

def extract_wrong_predictions():
    with open('common_misses.txt', 'r') as f:
        lines = f.readlines()
    df = pd.DataFrame(columns=['item_name', 'text', 'label', 'pred'])
    for line in lines:
        dataset, index = line.strip().split('-')
        if shot_type < 1:
            result_folder = os.path.join('../results', dataset, 'vicuna/zero-shot/vicuna-7')
            if dataset == 'jira':
                result_folder = os.path.join('../results', dataset, 'vicuna/zero-shot/vicuna-jira-7')
        else:
            result_folder = os.path.join('../results', dataset, 'vicuna/few-shot/vicuna/5')
            if dataset == 'jira':
                result_folder = os.path.join('../results', dataset, 'vicuna/few-shot/vicuna-jira/5')
                
        test_csv = os.path.join(result_folder, 'prediction.csv')
        test_df = pd.read_csv(test_csv)
        sentence = test_df.iloc[int(index)]['sentence']
        label = test_df.iloc[int(index)]['label']
        pred = test_df.iloc[int(index)]['pred']
        if dataset in ['github', 'so', 'app']:
            if label == 0:
                label = 'neutral'
            elif label == 1:
                label = 'positive'
            elif label == 2:
                label = 'negative'
            if pred == 0:
                pred = 'neutral'
            elif pred == 1:
                pred = 'positive'
            elif pred == 2:
                pred = 'negative'
        elif dataset in ['jira']:
            if label == 0:
                label = 'negative'
            elif label == 1:
                label = 'positive'
            if pred == 0:
                pred = 'negative'
            elif pred == 1:
                pred = 'positive'
        elif dataset in ['code']:
            if label == 0:
                label = 'non-negative'
            elif label == 1:
                label = 'negative'
            if pred == 0:
                pred = 'non-negative'
            elif pred == 1:
                pred = 'negative'
        df = df.append({'item_name': line.strip(), 'sentence': sentence, 'label': label, 'pred': pred}, ignore_index=True)
    df.to_csv('manual_check.csv', index=False)
    
def jira_neutral():
    for model in model_list:
        if shot_type < 1:
            result_folder = os.path.join('../results/jira/{}/zero-shot/{}-jira'.format(model, model_template_dict[model]))
            shot_candidates = [1, 2, 7]
        else:
            result_folder = os.path.join('../results/jira/{}/few-shot/{}-jira'.format(model, model_template_dict[model]))
            shot_candidates = [1, 3, 5]
        for shot in shot_candidates:
            for res_txt in os.listdir(result_folder):
                if res_txt.endswith('.txt'):
                    with open(os.path.join(result_folder, res_txt), 'r') as f:
                        lines = f.readlines()
                    

        
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--shot', '-s', type=int, required=True)
    args = args.parse_args()
    shot_type = args.shot
    
    # analyze_few_zero_misses()
    extract_wrong_predictions()