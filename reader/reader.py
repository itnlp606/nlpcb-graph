import os
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm
from utils.constants import *
from transformers import AutoTokenizer
from collections import defaultdict
from utils.utils import print_execute_time, print_empty_line
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def ner_tensorize(data, tokenizer, args, mode='seq'):
    # divide into tags and texts
    if mode == 'seq':    
        _, _, all_tokenized_sents, _, _, all_labels = ner_preprocess(data, tokenizer)
        dataset = TensorDataset(all_tokenized_sents['input_ids'], all_tokenized_sents['attention_mask'],\
            all_tokenized_sents['offset_mapping'], all_labels)
        sampler = SequentialSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    elif mode == 'random':
        pos_tokenized_sents, neg_tokenized_sents, _, pos_labels, neg_labels,\
            _ = ner_preprocess(data, tokenizer)
        
        # construct data loader
        pos_dataset = TensorDataset(pos_tokenized_sents['input_ids'], pos_tokenized_sents['attention_amask'],\
            pos_tokenized_sents['offset_mapping'], pos_labels)        
        neg_dataset = TensorDataset(neg_tokenized_sents['input_ids'], neg_tokenized_sents['attention_amask'],\
            neg_tokenized_sents['offset_mapping'], neg_labels)
        pos_sampler, neg_sampler = RandomSampler(pos_dataset), RandomSampler(neg_dataset)
        pos_loader = DataLoader(pos_dataset, sampler=pos_sampler, batch_size=3)
        neg_loader = DataLoader(neg_dataset, sampler=neg_sampler, batch_size=5)
        return pos_loader, neg_loader

# return tokenized_data, labels
def ner_preprocess(data, tokenizer):
    sents, labs = [], []
    for sent, lab in data:
        sents.append(sent)
        tt = [NER_LABEL2ID['O'] for _ in range(len(sent))]
        for tup in lab:
            start, end, word = tup
            # if sent[start:end] != word:
            #     raise Exception('wrong match')
            tt[start] = NER_LABEL2ID['B']
            for i in range(start+1, end):
                tt[i] = NER_LABEL2ID['I']
        if lab == []:
            labs.append((tt, 'neg'))
        else:
            labs.append((tt, 'pos'))

    all_sents, all_labels = [], []
    pos_sents, neg_sents, pos_labels, neg_labels = [], [], [], []
    for sent, (tt, tag) in zip(sents, labs):
        tokenized_sent = tokenizer(sent, return_offsets_mapping=True)
        label = []
        for idx, mp in enumerate(tokenized_sent['offset_mapping']):
            if idx > 0 and mp[0] == 0 and mp[1] == 0:
                break
            else:
                label.append(tt[mp[0]])
        if tag == 'pos':
            pos_sents.append(sent)
            pos_labels.append(label)
        else:
            neg_sents.append(sent)
            neg_labels.append(label)
        all_sents.append(sent)
        all_labels.append(label)

    pos_tokenized_sents = tokenizer(pos_sents, padding=True, truncation=True,\
        return_offsets_mapping=True, return_tensors='pt')
    neg_tokenized_sents = tokenizer(neg_sents, padding=True, truncation=True,\
        return_offsets_mapping=True, return_tensors='pt')
    all_tokenized_sents = tokenizer(all_sents, padding=True, truncation=True,\
        return_offsets_mapping=True, return_tensors='pt')
    pos_seq_len, neg_seq_len, all_seq_len = pos_tokenized_sents['input_ids'].shape[1], \
        neg_tokenized_sents['input_ids'].shape[1], all_tokenized_sents['input_ids'].shape[1]

    for label in pos_labels:
        label.extend([0]*(pos_seq_len - len(label)))
    for label in neg_labels:
        label.extend([0]*(neg_seq_len - len(label)))
    for label in all_labels:
        label.extend([0]*(all_seq_len - len(label)))

    return pos_tokenized_sents, neg_tokenized_sents, all_tokenized_sents,\
        torch.tensor(pos_labels), torch.tensor(neg_labels), torch.tensor(all_labels)

def clas_tensorize(data, tokenizer, args, mode='seq'):
    # divide into tags and texts
    tokenized_data, labels = clas_preprocess(data, tokenizer)
    if mode == 'seq':
        ids, masks = tokenized_data['input_ids'], tokenized_data['attention_mask']
        dataset = TensorDataset(ids, masks, labels)
        sampler = SequentialSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
    elif mode == 'random':
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
        
        # construct data loader
        pos_dataset = TensorDataset(pos_ids, pos_masks, torch.ones(pos_ids.shape[0], dtype=torch.int64))
        neg_dataset = TensorDataset(neg_ids, neg_masks, torch.zeros(neg_ids.shape[0], dtype=torch.int64))
        pos_sampler, neg_sampler = RandomSampler(pos_dataset), RandomSampler(neg_dataset)
        pos_loader = DataLoader(pos_dataset, sampler=pos_sampler, batch_size=3)
        neg_loader = DataLoader(neg_dataset, sampler=neg_sampler, batch_size=5)
        return pos_loader, neg_loader

# return tokenizer, labels
def clas_preprocess(data, tokenizer):
    labels, texts = [], []
    for tup in data:
        text, label = tup
        labels.append(int(label))
        texts.append(text)
    tokenized_text = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    labels = torch.tensor(labels)
    
    return tokenized_text, labels

@print_execute_time
def data2numpy():
    tasks = os.listdir('data')
    clas_array = []
    ner_array = []
    mx_len = 0

    dd = {0:0, 1:0, 2:0}
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', \
    #     cache_dir='pretrained_models', use_fast=True)

    for task in tasks:
        # ignore readme
        if task[-3:] == '.md':
            continue

        # process
        articles = os.listdir('data/'+task)
        pattern = 'Stanza-out.txt'
        paragraph_pat = 'Grobid-out.txt'
        sent_pat = 'sentences.txt'
        entity_pat = 'entities.txt'

        for article in articles:
            # 提取句子文件
            files = os.listdir('data/'+task+'/'+article)
            for f in files:
                if pattern in f:
                    name = f
                if paragraph_pat in f:
                    para_name = f
            
            # get dir, data
            para_dir = 'data/'+task+'/'+article+'/'+para_name
            sent_dir = 'data/'+task+'/'+article+'/'+name
            label_dir = 'data/'+task+'/'+article+'/'+sent_pat
            entity_dir = 'data/'+task+'/'+article+'/'+entity_pat
            with open(para_dir, 'r') as f:
                paras = f.readlines()
            with open(sent_dir, 'r') as f:
                sents = f.readlines()
            with open(label_dir, 'r') as f:
                labels = f.readlines()
            with open(entity_dir, 'r') as f:
                entities = f.readlines()
            labels = [int(label) for label in labels]
            
            def get_context(i):
                if i < 0 or i >= len(sents):
                    return ""
                return '#' + sents[i]

            # process entities
            sentID2entites = defaultdict(list)
            for entity in entities:
                if entity[-1] == '\n': entity = entity[:-1]
                items = entity.split('\t')
                sentID2entites[int(items[0])].append((\
                    int(items[1]), int(items[2]), items[3]))

            # append data
            for i, sent in enumerate(sents):
                # 去掉回车
                if sent[-1] == '\n':
                    sent = sent[:-1]

                # 加特征前处理NER问题
                if i+1 not in labels:
                    ner_array.append((sent, []))
                else:
                    ner_array.append((sent, sentID2entites[i+1]))

                # add feature
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

                K = 1
                for cxt in range(K):
                    sent += get_context(i-K) + get_context(i+K)

                # s = tokenizer(sent)
                # if len(s['input_ids']) < 510:
                #     dd[0] += 1
                # else: 
                #     dd[1] += 1
                #     if i+1 in labels:
                #         dd[2] += 1

                if i+1 in labels:
                    clas_array.append((sent, 1))#label_id))
                else:
                    clas_array.append((sent, 0))
    
    # with open('array.pkl', 'wb') as f:
    #     pickle.dump(np.array(array), f)

    # print(dd)
    # raise Exception

    return np.array(clas_array), np.array(ner_array, dtype=object)