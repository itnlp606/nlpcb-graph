import os
import json
import torch
import pickle
import numpy as np
from utils.constants import *
from transformers import AutoTokenizer
from collections import defaultdict
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
            if label >= 1:
                pos_ids.append(ids)
                pos_masks.append(mask)
            else:
                neg_ids.append(ids)
                neg_masks.append(mask)
        
        pos_ids, pos_masks, neg_ids, neg_masks = torch.stack(pos_ids), \
            torch.stack(pos_masks), torch.stack(neg_ids), torch.stack(neg_masks)

        # upsampling
        # mul = neg_ids.shape[0] // pos_ids.shape[0]
        # pos_ids = torch.stack([pos_ids]*mul).view(-1, pos_ids.shape[-1])
        # pos_masks = torch.stack([pos_masks]*mul).view(-1, pos_masks.shape[-1])
        # pos_labels = torch.stack([pos_labels]*mul).view(-1, pos_labels.shape[-1]).squeeze(1)
        
        # construct data loader
        pos_dataset = TensorDataset(pos_ids, pos_masks, torch.ones(pos_ids.shape[0], dtype=torch.int64))
        neg_dataset = TensorDataset(neg_ids, neg_masks, torch.zeros(neg_ids.shape[0], dtype=torch.int64))
        pos_sampler, neg_sampler = RandomSampler(pos_dataset), RandomSampler(neg_dataset)
        pos_loader = DataLoader(pos_dataset, sampler=pos_sampler, batch_size=2)
        neg_loader = DataLoader(neg_dataset, sampler=neg_sampler, batch_size=6)
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

@print_execute_time
def data2numpy():
    tasks = os.listdir('data')
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', \
    #     cache_dir='pretrained_models', use_fast=True)
    array = []
    mx_len = 0

    dd = {0:0, 1:0}

    for task in tasks:
        # ignore readme
        if task[-3:] == '.md':
            continue

        # process
        articles = os.listdir('data/'+task)
        pattern = 'Stanza-out.txt'
        paragraph_pat = 'Grobid-out.txt'
        sent_pat = 'sentences.txt'

        for article in articles:
            # 提取句子文件
            files = os.listdir('data/'+task+'/'+article)
            for f in files:
                if pattern in f:
                    name = f
                if paragraph_pat in f:
                    para_name = f

            # # 构建句子id-实体 词典
            # sentid2entities = defaultdict(list)
            # with open('data/'+task+'/'+article+'/entities.txt') as f:
            #     content = f.readlines()

            # for line in content:
            #     line = line.split('\t')
            #     sent_id, entity = int(line[0]), line[-1]
            #     if entity[-1] == '\n': entity = entity[:-1]
            #     sentid2entities[sent_id].append(entity)

            # # 类别名称-三元组文件字典
            # label2triples = {}
            # files = os.listdir('data/'+task+'/'+article+'/triples')
            # for label in files:
            #     with open('data/'+task+'/'+article+'/triples/'+label) as f:
            #         label2triples[label[:-4]] = f.read()
            
            # get dir, data
            para_dir = 'data/'+task+'/'+article+'/'+para_name
            sent_dir = 'data/'+task+'/'+article+'/'+name
            label_dir = 'data/'+task+'/'+article+'/'+sent_pat
            with open(para_dir, 'r') as f:
                paras = f.readlines()
            with open(sent_dir, 'r') as f:
                sents = f.readlines()
            with open(label_dir, 'r') as f:
                labels = f.readlines()
            labels = [int(label) for label in labels]

            # append data
            for i, sent in enumerate(sents):
                # add feature
                if sent[-1] == '\n':
                    sent = sent[:-1]

                # extract title
                for j, para in enumerate(paras):
                    if sent in para:
                        idx = j

                while idx >= 0 and paras[idx] != '\n': idx -= 1
                if idx == 0: title = paras[0]
                else: title = paras[idx+1]

                if title[-1] == '\n':
                    title = title[:-1]
                
                sent += '#' + title

                if i == 0:
                    sent += '#' + sents[1]
                elif i == len(sents)-1:
                    sent += '#' + sents[-2]
                else:
                    sent += '#' + sents[i-1] + '#' + sents[i-1]

                # s = tokenizer(sent)
                # if len(s['input_ids']) < 510:
                #     dd[0] += 1
                # else: dd[1] += 1

                if i+1 in labels:
                    # get label_id
                    # entities = sentid2entities[i+1]

                    # vote_box = [0 for _ in range(len(LABEL2ID)+1)]
                    # for entity in entities:
                    #     for idx, label in enumerate(label2triples):
                    #         if entity in label2triples[label]:
                    #             vote_box[LABEL2ID[label]] += 1
                    # label_id = np.argmax(vote_box)

                    # assign class
                    array.append((sent, 1))#label_id))
                else:
                    array.append((sent, 0))
    
    # with open('array.pkl', 'wb') as f:
    #     pickle.dump(np.array(array), f)

    # print(dd)
    raise Exception

    return np.array(array)