import pandas as pd
import icecream as ic
from tqdm import tqdm
import os

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

general_score_polarity_dict = {
    '2': 'negative',
    '0': 'neutral',
    '1': 'positive'
}

code_score_polarity_dict = {
    '1': 'negative',
    '0': 'non-negative'
}

jira_score_polarity_dict = {
    '0': 'negative',
    '1': 'positive'
}



def generate_few_shots(dataset, prompt_template, shot_type):
    shot_folder = '../sampled-shot/{}-shot/{}'.format(shot_type, dataset)
    os.makedirs(shot_folder, exist_ok=True)
    
    train_csv = '../data/{}-train.csv'.format(dataset)
    train_df = pd.read_csv(train_csv)
    
    if dataset == 'app':
        test_csv = '../data/{}-test.csv'.format(dataset)
    else:
        test_csv = '../data/{}-sampled-test.csv'.format(dataset)
    test_df = pd.read_csv(test_csv)
    
    if dataset == 'code':
        cur_polarity_dict = code_score_polarity_dict
    elif dataset == 'jira':
        cur_polarity_dict = jira_score_polarity_dict
    else:
        cur_polarity_dict = general_score_polarity_dict
    
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        if os.path.exists(os.path.join(shot_folder, '{}.txt'.format(i))):
            continue
        sampled_examples = train_df.sample(n=shot_type)
        prompt_list = []
        for j in range(shot_type):
            prompt_list.append(sampled_examples.iloc[j]['sentence'])
            prompt_list.append(cur_polarity_dict[str(sampled_examples.iloc[j]['label'])])
        prompt_list.append(row['sentence'])
        prompt = prompt_template.format(*prompt_list)
        with open(os.path.join(shot_folder, '{}.txt'.format(i)), 'w') as f:
            f.write(prompt)
        
                
if __name__ == '__main__':
    template_dict ={
       "1": "Example Sentence: {}\nLabel: {}\n",
       "3": "Example Sentence: {}\nLabel: {}\nExample Sentence: {}\nLabel: {}\nExample Sentence: {}\nLabel: {}\n",
       "5": "Example Sentence: {}\nLabel: {}\nExample Sentence: {}\nLabel: {}\nExample Sentence: {}\nLabel: {}\nExample Sentence: {}\nLabel: {}\nExample Sentence: {}\nLabel: {}\n"
    }
    
    for dataset in ['github', 'so', 'app', 'jira', 'code']:
        for shot_type in [1, 3, 5]:
            prompt_template = template_dict[str(shot_type)]
            generate_few_shots(dataset, prompt_template, shot_type)