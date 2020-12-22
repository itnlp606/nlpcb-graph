import os
import json
import torch
import pickle
import numpy as np
from utils.constants import *
from utils.utils import print_execute_time, print_empty_line
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def tensorize(data, tokenizer, args, mode='seq'):
    # divide into tags and texts
    if mode == 'seq':
        tokenized_data, labels = preprocess(data, tokenizer)
        ids, masks = tokenized_data['input_ids'], tokenized_data['attention_mask']
        dataset = TensorDataset(ids, masks, labels)
        sampler = SequentialSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
    elif mode == 'random':
        tokenized_pos, tokenized_neg, pos_labels, neg_labels = negsamp_preprocess(data, tokenizer)
        pos_dataset = TensorDataset(tokenized_pos['input_ids'], tokenized_pos['attention_mask'], pos_labels)
        neg_dataset = TensorDataset(tokenized_neg['input_ids'], tokenized_neg['attention_mask'], neg_labels)
        pos_sampler, neg_sampler = RandomSampler(pos_dataset), RandomSampler(neg_dataset)
        pos_loader = DataLoader(pos_dataset, sampler=pos_sampler, batch_size=4)
        neg_loader = DataLoader(neg_dataset, sampler=neg_sampler, batch_size=12)
        return pos_loader, neg_loader

def negsamp_preprocess(data, tokenizer):
    # negative sampling
    pos_data, pos_labels, neg_data, neg_labels = [], [], [], []

    for tup in data:
        text, label = tup
        if label == '1':
            pos_data.append(text)
            pos_labels.append(int(label))
        else:
            neg_labels.append(int(label))
            neg_data.append(text)
    tokenized_pos = tokenizer(pos_data, padding='max_length', truncation=True, return_tensors='pt')
    tokenized_neg = tokenizer(neg_data, padding='max_length', truncation=True, return_tensors='pt')
    pos_labels = torch.tensor(pos_labels)
    neg_labels = torch.tensor(neg_labels)

    return tokenized_pos, tokenized_neg, pos_labels, neg_labels

# return tokenizer, labels
def preprocess(data, tokenizer):
    labels, texts = [], []
    for tup in data:
        text, label = tup
        labels.append(int(label))
        texts.append(text)
    tokenized_text = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    labels = torch.tensor(labels)

    return tokenized_text, labels

def divide_dataset(array, num_fold, fold):
    np.random.seed(2233)
    np.random.shuffle(array)
    block_len = int(array.shape[0]/10)

    num_blocks = num_fold-fold
    start = block_len*num_blocks
    a = np.vstack([array[0:start], array[start+block_len:]])
    return a, array[start:start+block_len]

def data2numpy():
    tasks = os.listdir('data')
    array = []
    mx_len = 0
    for task in tasks:
        # ignore readme
        if task[-3:] == '.md':
            continue

        # process
        articles = os.listdir('data/'+task)
        pattern = 'Stanza-out.txt'
        sent_pat = 'sentences.txt'

        for article in articles:
            files = os.listdir('data/'+task+'/'+article)
            for f in files:
                if pattern in f:
                    name = f
                    break
            
            # get dir, data
            sent_dir = 'data/'+task+'/'+article+'/'+name
            label_dir = 'data/'+task+'/'+article+'/'+sent_pat
            with open(sent_dir, 'r') as f:
                sents = f.readlines()
            with open(label_dir, 'r') as f:
                labels = f.readlines()
            labels = [int(label) for label in labels]

            # append data
            for i, sent in enumerate(sents):
                if i+1 in labels:
                    array.append((sent, 1))
                else:
                    array.append((sent, 0))
    
    with open('array.pkl', 'wb') as f:
        pickle.dump(np.array(array), f)

    return np.array(array)