from transformers import AdamW
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from transformers import get_scheduler
import copy
import pandas as pd
from logger import init_logger
import os
import datetime
import logging
from utils import seed_everything
import argparse
from torch.utils.data import Dataset

lr = 2e-5
bs = 32
seed = 42
max_len = 256
num_epochs = 5


name_tokenizer_model = {
    'BERT': 'bert-base-uncased',
    'RoBERTa': 'roberta-base',
    'XLNet': "xlnet-base-cased",
    'ALBERT': 'albert-base-v2',
    'distilbert': 'distilbert-base-uncased'
}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CustomizedDataset(Dataset):
    def __init__(self, data, labels, model):
        self.data = data
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = str(self.data[index])

        encoded = self.tokenizer(
            sentence,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        token_ids = encoded['input_ids'].squeeze(0)
        attn_masks = encoded['attention_mask'].squeeze(0)

        label = self.labels[index]
        return token_ids, attn_masks, label  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--variant', '-v', help='BERT variant')
    parser.add_argument('--dataset', '-d', help='dataset')
    args = parser.parse_args()
    
    seed_everything(seed)
    variant = args.variant
    dataset = args.dataset
    num_labels = 3
    label_list = [0, 1, 2]
    if dataset == 'jira' or dataset == 'code':
        num_labels = 2
        label_list = [0, 1]
        
    init_logger('../log/{}_{}_sampled_{}.log'.format(variant, dataset, datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))
    logging.info('learning rate is :{}, max_len is {}'.format(lr, max_len))
    logging.info('seed is {}'.format(seed))
    logging.info('batch size is {}'.format(bs))

    split_dir = '../data/'

    logging.info('data folder is {}'.format(split_dir))
    logging.info('running {}'.format(variant))
    
    all_f1 = []
    all_precision = []
    all_recall = []
    all_accuracy = []
    
    best_f1 = 0
    
    os.makedirs('../model/{}/'.format(dataset), exist_ok=True)
    model_save_path = '../model/{}/{}_sampled_{}.pt'.format(dataset, variant, seed)
    train_df = pd.read_csv('../data/{}-train.csv'.format(dataset))
    valid_df = pd.read_csv('../data/{}-valid.csv'.format(dataset))
    if dataset == 'app':
        test_df = pd.read_csv('../data/{}-test.csv'.format(dataset))
    else:
        test_df = pd.read_csv('../data/{}-sampled-test.csv'.format(dataset))
    
    train_texts, train_labels = train_df['sentence'], train_df['label']
    valid_texts, valid_labels = valid_df['sentence'], valid_df['label']
    test_texts, test_labels = test_df['sentence'], test_df['label']
    
    tokenizer = AutoTokenizer.from_pretrained(name_tokenizer_model[variant], do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(name_tokenizer_model[variant], num_labels = num_labels)

    train_dataset = CustomizedDataset(train_texts, train_labels, name_tokenizer_model[variant])
    val_dataset = CustomizedDataset(valid_texts, valid_labels, name_tokenizer_model[variant])
    test_dataset = CustomizedDataset(test_texts, test_labels, name_tokenizer_model[variant])

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=bs)
    eval_dataloader = DataLoader(val_dataset, batch_size=bs)
    test_dataloader = DataLoader(test_dataset, batch_size=bs)

    model.to(device)
    num_training_steps = num_epochs * len(train_dataloader)

    optimizer = AdamW(model.parameters(), lr=lr)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))
    best_loss = np.Inf
    model_copy = 0
    
    for epoch in range(num_epochs):
        logging.info('Epoch: {}'.format(epoch))
        logging.info('starting to train the model....')
        
        model.train()
        for batch in train_dataloader:
            seq, attn_masks, labels = batch
            seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)

            outputs = model(input_ids=seq, attention_mask=attn_masks, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        logging.info('starting to evaluate the model....')

        val_loss = []

        model.eval()
        all_predictions = []
        all_labels = []
        
        for batch in eval_dataloader:
            seq, attn_masks, labels = batch
            seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(input_ids=seq, attention_mask=attn_masks, labels=labels)
                
            val_loss.append(outputs.loss.item())
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).flatten()
            pred = predictions.cpu()
            
            truth = labels.cpu()
            all_predictions.extend(pred)
            all_labels.extend(truth)
            
        val_loss = np.mean(val_loss)
        cur_accuracy = accuracy_score(all_labels, all_predictions)
        cur_f1 = f1_score(all_labels, all_predictions, labels=label_list, average='macro')
        cur_precision = precision_score(all_labels, all_predictions, labels=label_list, average='macro')
        cur_recall = recall_score(all_labels, all_predictions, labels=label_list, average='macro')
        
        logging.info('the result on validation data....')
        logging.info('accuracy: {}'.format(cur_accuracy))
        logging.info('recall: {}'.format(cur_recall))
        logging.info('precision: {}'.format(cur_precision))
        logging.info('f1: {}'.format(cur_f1))
    
        # if val_loss < best_loss:
        #     logging.info('saving the best loss, changed from {} to {}'.format(best_loss, val_loss))
        #     best_loss = val_loss
        #     # torch.save(model.state_dict(), model_save_path)
        #     model_copy = copy.deepcopy(model)

        if cur_f1 >= best_f1:
            logging.info('saving the best f1, changed from {} to {}'.format(best_f1, cur_f1))
            best_f1 = cur_f1
            model_copy = copy.deepcopy(model)
        
    # logging.info('overall the best loss is {}'.format(best_loss))
    logging.info('overall the best f1 is {}'.format(best_f1))
    torch.save(model_copy.state_dict(), model_save_path)
    del loss
    torch.cuda.empty_cache()

    checkpoint = torch.load(model_save_path)
    model = AutoModelForSequenceClassification.from_pretrained(name_tokenizer_model[variant], num_labels = num_labels)

    model.load_state_dict(checkpoint)
    model.cuda()
    logging.info('succefully loaded')

    model.eval()
    cur_accuracy = []
    test_loss = []
    cur_precision = []
    cur_recall = []
    cur_f1 = []
    
    all_predictions, all_labels = [], []

    for batch in test_dataloader:
        seq, attn_masks, labels = batch
        seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids=seq, attention_mask=attn_masks, labels=labels)
            
        test_loss.append(outputs.loss.item())
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).flatten()

        pred = predictions.cpu()
        truth = labels.cpu()
        all_predictions.extend(pred)
        all_labels.extend(truth)


    # cur_accuracy = np.mean(cur_accuracy)
    cur_accuracy = accuracy_score(all_labels, all_predictions)
    cur_f1 = f1_score(all_labels, all_predictions, labels=label_list, average='macro')
    cur_precision = precision_score(all_labels, all_predictions, labels=label_list, average='macro')
    cur_recall = recall_score(all_labels, all_predictions, labels=label_list, average='macro')
    
    logging.info('------- the result on test data -------')
    logging.info('accuracy: {}'.format(cur_accuracy))
    logging.info('recall: {}'.format(cur_recall))
    logging.info('precision: {}'.format(cur_precision))
    logging.info('f1: {}'.format(cur_f1))