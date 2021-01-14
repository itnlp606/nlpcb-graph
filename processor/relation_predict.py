import os
import torch
import numpy as np
import pandas as pd
import transformers
from time import time
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations
from model.bert import BERTRE
from collections import defaultdict
from utils.utils import print_execute_time
from utils.constants import *
from collections import defaultdict
from processor.preprocessor import clas_tensorize
from utils.utils import divide_dataset
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

def relation_predict(args, tokenizer, device, data_folder):
    vote_knife = 4
    tt_nums = 0
    tt_trips = 0

    base_dir = 'relation_models'
    mods = os.listdir(base_dir)

    tasks = os.listdir(data_folder)
    for task in tasks:
        print(task)
        # ignore readme
        if task[-3:] == '.md' or task[-4:] == '.git' or task[-4:] == '.zip':
            continue

        # if task != 'natural_language_inference': continue

        task_path = 'results/'+task

        # process
        articles = os.listdir(data_folder+'/'+task)
        pattern = 'Stanza-out.txt'
        paragraph_pat = 'Grobid-out.txt'

        for article in tqdm(articles):
            # mkdir
            trip_path = task_path + '/' + article + '/triples'

            if not os.path.exists(trip_path):
                os.mkdir(trip_path)

            result = defaultdict(list)

            # 提取句子文件
            files = os.listdir(data_folder+'/'+task+'/'+article)
            for f in files:
                if pattern in f:
                    name = f
            
            # get dir, data
            sent_dir = data_folder+'/'+task+'/'+article+'/'+name
            
            with open(sent_dir, 'r') as f:
                sents = f.readlines()

            with open('results/'+task+'/'+article+'/entities.txt', 'r') as f:
                entities = f.readlines()

            # 预处理实体
            sent2entities = defaultdict(list)
            for ent in entities:
                tup = ent.strip().split('\t')
                tup[0], tup[1], tup[2] = int(tup[0]), int(tup[1]), int(tup[2])
                sent2entities[tup[0]].append((tup[1], tup[2], tup[3]))

            inputs = []
            # 变成统一形式
            for sent_id in sent2entities:
                sent = sents[sent_id-1]

                # 不用去回车，因为预处理就没去掉
                ents = sent2entities[sent_id]

                # 仅考虑两种特殊情况
                t1 = '#Contribution#has research problem#'
                t2 = '#Contribution#Code#'
                if len(ents) < 3:
                    for ent in ents:
                        inputs.append(sent+t1+ent[2])
                        inputs.append(sent+t2+ent[2])

                # 考虑所有排列组合
                else:
                    combs = combinations(ents, 3)
                    for comb in combs:
                        sample = deepcopy(sent)
                        for ent in comb:
                            sample += '#' + ent[2]
                        inputs.append(sample)

            tokenized_sents = tokenizer(inputs, padding=True, truncation=True, \
                return_tensors='pt')
            labels = torch.zeros((tokenized_sents['input_ids'].shape[0], 1), dtype=torch.int64)
            dataset = TensorDataset(tokenized_sents['input_ids'], tokenized_sents['attention_mask'], \
                labels)
            loader = DataLoader(dataset, args.batch_size)

            with torch.no_grad():
                for q, data in enumerate(loader):
                    data = tuple(i.to(device) for i in data)
                    ids, masks, labels = data
                    
                    vote_box = {}
                    for mod in mods:
                        model = torch.load(base_dir+'/'+mod, map_location=torch.device('cpu'))
                        model = model.to(device)
                        model.eval()
                        _, logits = model(ids, masks, labels)
                        logits = torch.argmax(logits, 1)

                        for idx, logit in enumerate(logits):
                            if idx not in vote_box:
                                vote_box[idx] = [0 for _ in range(len(ID2BLOCK)+1)]
                            vote_box[idx][logit] += 1
                    
                    # 放入打印区
                    for idx in vote_box:
                        input0 = inputs[q*args.batch_size + idx]
                        t = np.argmax(vote_box[idx])
                        if t > 0 and vote_box[idx][t] >= vote_knife:
                            # 提取三元组
                            tup = input0.split('#')

                            # 放入待打印区
                            result[t].append((tup[1], tup[2], tup[3]))

            tt_nums += 1
            # 写进文件

            for id0 in result:
                tt_trips += len(result[id0])
                name = BLOCK2FILENAME[ID2BLOCK[id0]]
                with open(trip_path+'/'+name, 'w') as f:
                    triples = result[id0]
                    for triple in triples:
                        if ID2BLOCK[id0] == 'Code' and triple[1] != 'Code': continue
                        if ID2BLOCK[id0] == 'has research problem' and triple[1] != 'has research problem': continue
                        f.write('('+triple[0]+'||'+triple[1]+'||'+triple[2]+')\n')

    print(tt_trips/tt_nums, tt_trips, tt_nums)