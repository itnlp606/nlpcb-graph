import os
import torch
import pandas as pd
import transformers
from time import time
from tqdm import tqdm
from model.bert import BERTNER
from collections import defaultdict
from utils.utils import print_execute_time
from processor.preprocessor import clas_tensorize
from utils.utils import divide_dataset
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

def ner_predict(args, tokenizer, device, data_folder):
    vote_knife = 5
    tt_nums = 0
    tt_ents = 0

    base_dir = 'ner_models'
    mods = os.listdir(base_dir)

    tasks = os.listdir(data_folder)
    for task in tasks:
        # ignore readme
        if task[-3:] == '.md' or task[-4:] == '.git' or task[-4:] == '.zip':
            continue

        if task != 'natural_language_inference': continue

        task_path = data_folder+'/'+task

        # process
        articles = os.listdir(data_folder+'/'+task)
        pattern = 'Stanza-out.txt'
        paragraph_pat = 'Grobid-out.txt'

        for article in tqdm(articles):
            # mkdir
            path = task_path + '/' + article
            if not os.path.exists(path):
                os.mkdir(path)

            result = []

            # 提取句子文件
            files = os.listdir(data_folder+'/'+task+'/'+article)
            for f in files:
                if pattern in f:
                    name = f
            
            # get dir, data
            sent_dir = data_folder+'/'+task+'/'+article+'/'+name
            
            with open(sent_dir, 'r') as f:
                sents = f.readlines()

            with open('results/'+task+'/'+article+'/sentences.txt', 'r') as f:
                right_sents_idx = f.readlines()

            # 预处理索引
            for i in range(len(right_sents_idx)):
                if right_sents_idx[i][-1] == '\n': 
                    right_sents_idx[i] = right_sents_idx[i][:-1]
                right_sents_idx[i] = int(right_sents_idx[i])

            right_sents = []
            for sent in right_sents_idx:
                s = sents[sent-1]
                if s[-1] == '\n': s = s[:-1]
                right_sents.append(s)

            tokenized_sents = tokenizer(right_sents, padding=True, truncation=True, \
                return_offsets_mapping=True, return_tensors='pt')
            labels = torch.zeros(tokenized_sents['input_ids'].shape, dtype=torch.int64)
            dataset = TensorDataset(tokenized_sents['input_ids'], tokenized_sents['attention_mask'], \
                tokenized_sents['offset_mapping'], labels)
            loader = DataLoader(dataset, args.batch_size)

            with torch.no_grad():
                vote_box = defaultdict(dict)
                for q, data in enumerate(loader):
                    data = tuple(i.to(device) for i in data)
                    ids, masks, maps, labels = data
                    
                    for mod in mods:
                        model = torch.load(base_dir+'/'+mod, map_location=torch.device('cpu'))
                        model = model.to(device)
                        model.eval()
                        _, logits = model(ids, masks, labels)

                        # 处理logits
                        for jdx, (seq, mp, mask) in enumerate(zip(logits, maps, masks)):
                            # print(seq)
                            i = 0
                            ints = []
                            while i < len(seq):
                                if mask[i] == 0: break
                                j = i+1
                                if seq[i] == 1:
                                    while j < len(seq) and seq[j] == 2:
                                        j += 1
                                    # i, j转map

                                    # print(i, j, mp[i][0].item(), mp[j-1][1].item())
                                    ints.append((mp[i][0].item(), mp[j-1][1].item()))
                                i = j

                            for it in ints:
                                if it in vote_box[q*args.batch_size+jdx]:
                                    vote_box[q*args.batch_size+jdx][it] += 1
                                else:
                                    vote_box[q*args.batch_size+jdx][it] = 1

                # 处理vote_box
                ents_to_write = [] # 句子id，实体itv，实体名称
                for key in vote_box:
                    art = right_sents_idx[key]
                    target_sent = right_sents[key]
                    intv2num = vote_box[key]
                    # print(intv2num)
                    for intv in intv2num:
                        if intv2num[intv] >= vote_knife:
                            ents_to_write.append(str(art)+'\t'+str(intv[0])+'\t'+\
                                str(intv[1])+'\t'+target_sent[intv[0]:intv[1]])

            tt_ents += len(ents_to_write)
            tt_nums += 1
            # 写进文件
            with open('results/'+task+'/'+article+'/entities.txt', 'w') as f:
                for idx, s in enumerate(ents_to_write):
                    if idx == 0: f.write(s)
                    else: f.write('\n'+s)

    print(tt_ents/tt_nums, tt_ents, tt_nums)