import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
sample_dict = {
    'app': [33, 35],
    'code': [114, 160],
    'github': [250, 713],
    'jira': [76, 93],
    'so': [109, 150]
}

def export_csv_code_review():
    """
    Convert xlsx to csv
    """
    
    raw_file_path = '../data/original/code-reviews.xlsx'
    df = pd.read_excel(raw_file_path, header=None)
    df.columns = ['sentence', 'label']
    df['label'] = df['label'].apply(lambda x: 1 if x == -1 else 0)
    print(df.head())
    df.to_csv('../data/code.csv', index=False)
    
def export_csv_app_reviews():
    raw_file_path = '../data/original/AppReviews.csv'
    df = pd.read_csv(raw_file_path)
    df.columns = ['sentence', 'label']
    print(df.shape)
    # -1: 2, 0: 0, 1: 1
    df['label'] = df['label'].apply(lambda x: 1 if x == 1 else (2 if x < 0 else 0))
    df.to_csv('../data/app.csv', index=False)
    
def export_csv_github():
    df = pd.read_csv('../data/original/github_gold.csv', usecols=['Polarity', 'Text'], sep=';')
    df.columns = ['label', 'sentence']
    df['label'] = df['label'].apply(lambda x: 1 if x == 'positive' else (2 if x == 'negative' else 0))
    print(df.shape)
    df.to_csv('../data/github.csv', index=False)
    
def export_csv_jira():
    df = pd.read_csv('../data/original/JIRA.csv', usecols=['sentence', 'oracle'])
    df.columns = ['sentence', 'label']
    df['label'] = df['label'].apply(lambda x: 0 if x == -1 else 1)
    print(df.shape)
    df.to_csv('../data/jira.csv', index=False)
    
def export_csv_stackoverflow():
    df = pd.read_csv('../data/original/StackOverflow.csv', usecols=['text', 'oracle'])
    df.columns = ['sentence', 'label']
    df['label'] = df['label'].apply(lambda x: 1 if x == 1 else (2 if x < 0 else 0))
    print(df.shape)
    df.to_csv('../data/so.csv', index=False)
    
def split_data(file_name):
    stratify_column = 'label'
    df = pd.read_csv(file_name)
    train_data, test_data = train_test_split(
        df, test_size=0.2, 
        stratify=df[stratify_column], random_state=42)
    valid_data, test_data = train_test_split(
        test_data, test_size=0.5,
        stratify=test_data[stratify_column], random_state=42)
    print(file_name)
    print(train_data.shape, valid_data.shape, test_data.shape)
    print('------------------' * 3)
    # print(valid_data[stratify_column].value_counts())
    # print(test_data[stratify_column].value_counts())
    train_data.to_csv(file_name.replace('.csv', '-train.csv'), index=False)
    valid_data.to_csv(file_name.replace('.csv', '-valid.csv'), index=False)
    test_data.to_csv(file_name.replace('.csv', '-test.csv'), index=False)
    
def sample_data(dataset):
    stratify_column = 'label'
    file_name = '../data/{}-test.csv'.format(dataset)
    test_size = sample_dict[dataset][0] / sample_dict[dataset][1]
    
    df = pd.read_csv(file_name)
    train_data, test_data = train_test_split(
        df, test_size= test_size,
        stratify=df[stratify_column], random_state=42
    )
    print(file_name)
    print(test_data.shape)
    print('------------------' * 3)
    test_data.to_csv('../data/{}-sampled-test.csv'.format(dataset), index=False)

def calculate_token_number(df):
    sentences = df['sentence'].tolist()
    tokens = []
    for sentence in sentences:
        tokens.append(len(sentence.split()))
    sum_tokens = sum(tokens)
    avg_tokens = sum_tokens / len(tokens)
    print('Average tokens: {}'.format(avg_tokens))
    
if __name__ == "__main__":
    # args = argparse.ArgumentParser()
    # args.add_argument('--dataset', '-d', type=str, required=True)
    # args = args.parse_args()
    # dataset = args.dataset
    
    # if dataset == 'github':
    #     export_csv_github()
    #     split_data('../data/github.csv')
    # elif dataset == 'jira':
    #     export_csv_jira()
    #     split_data('../data/jira.csv')
    # elif dataset == 'app':
    #     export_csv_app_reviews()
    #     split_data('../data/app.csv')
    # elif dataset == 'so':
    #     export_csv_stackoverflow()
    #     split_data('../data/so.csv')
    # elif dataset == 'code':
    #     export_csv_code_review()
    #     split_data('../data/code.csv')
    for dataset in ['github', 'jira', 'so', 'code', 'app']:
        # sample_data(dataset)
        df = pd.read_csv('../data/{}.csv'.format(dataset))
        print(dataset)
        # print(df['label'].value_counts())
        # print('------------------' * 3)
        calculate_token_number(df)