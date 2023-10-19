import pandas as pd
import icecream as ic

def check_polarity(file_name):
    df = pd.read_csv(file_name)
    df.columns = ['id', 'text', 'polarity']
    polarity_dict = {}
    for i in range(len(df)):
        cur_polarity = df.iloc[i]['polarity']
        if cur_polarity not in polarity_dict:
            polarity_dict[cur_polarity] = 1
        else:
            polarity_dict[cur_polarity] += 1
    for key, value in polarity_dict.items():
        print(f'{key}: {value}')
    
if __name__ == '__main__':
    check_polarity('../data/gh-test.csv')