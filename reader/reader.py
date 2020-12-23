import os
import json
import torch
import pickle
import numpy as np
from functools import reduce
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
        tokenized_data, labels = preprocess(data, tokenizer)
        pos_ids, pos_masks, neg_ids, neg_masks = [], [], [], []
        for i, (ids, mask, label) in enumerate(zip(tokenized_data['input_ids'], \
            tokenized_data['attention_mask'], labels)):
            ids = ids
            mask = mask
            if label == 1:
                pos_ids.append(ids)
                pos_masks.append(mask)
            else:
                neg_ids.append(ids)
                neg_masks.append(mask)
        
        pos_ids, pos_masks, neg_ids, neg_masks = torch.stack(pos_ids), \
            torch.stack(pos_masks), torch.stack(neg_ids), torch.stack(neg_masks)
        
        pos_dataset = TensorDataset(pos_ids, pos_masks, torch.ones(pos_ids.shape[0], dtype=torch.int32))
        neg_dataset = TensorDataset(neg_ids, neg_masks, torch.zeros(neg_ids.shape[0], dtype=torch.int32))
        pos_sampler, neg_sampler = RandomSampler(pos_dataset), RandomSampler(neg_dataset)
        pos_loader = DataLoader(pos_dataset, sampler=pos_sampler, batch_size=4)
        neg_loader = DataLoader(neg_dataset, sampler=neg_sampler, batch_size=12)
        return pos_loader, neg_loader

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