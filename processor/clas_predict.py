import os
import torch
import pandas as pd
import transformers
from time import time
from tqdm import tqdm
from model.bert import BERTCLAS, BERT
from utils.utils import print_execute_time
from processor.preprocessor import clas_tensorize
from utils.utils import divide_dataset
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

def clas_predict(args, tokenizer, device, data_folder):
    vote_knife = 5

    base_dir = 'sent_clas_models'
    mods = os.listdir(base_dir)

    tasks = os.listdir(data_folder)
    for task in tasks:
        # ignore readme
        if task[-3:] == '.md' or task[-4:] == '.git':
            continue

        task_path = 'results/'+task
        if not os.path.exists(task_path):
            os.mkdir(task_path)
        else: continue

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
                if paragraph_pat in f:
                    para_name = f
            
            # get dir, data
            sent_dir = data_folder+'/'+task+'/'+article+'/'+name
            para_dir = data_folder+'/'+task+'/'+article+'/'+para_name
            
            with open(sent_dir, 'r') as f:
                sents = f.readlines()

            with open(para_dir, 'r', encoding='ISO-8859-1') as f:
                paras = f.readlines()

            def get_context(i):
                if i < 0 or i >= len(sents):
                    return ""
                return '#' + sents[i]

            for i, sent in enumerate(sents):
                # 去掉回车
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
                
                K = 1
                for cxt in range(K):
                    sent += get_context(i-cxt) + get_context(i+cxt)
                sents[0] = sent

            labels = torch.tensor([0]*len(sents))
            tokenized_sents = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
            dataset = TensorDataset(tokenized_sents['input_ids'], tokenized_sents['attention_mask'], labels)
            loader = DataLoader(dataset, args.batch_size)

            start_time = time()
            for q, data in enumerate(loader):
                data = tuple(i.to(device) for i in data)
                ids, masks, labels = data
                
                vote_box = [0 for _ in range(args.batch_size)]
                for mod in mods:
                    model = torch.load(base_dir+'/'+mod, map_location=torch.device('cpu'))
                    model = model.to(device)
                    _, logits = model(ids, masks, labels)
                    logits = torch.argmax(logits, 1)
                    for i in range(len(logits)):
                        vote_box[i] += logits[i].item()

                for idx, vote in enumerate(vote_box):
                    if vote >= vote_knife:
                        result.append(q*args.batch_size + idx + 1)

            # 写进文件
            with open(path+'/sentences.txt', 'w') as f:
                for idx, sent in enumerate(result):
                    if idx == 0:
                        f.write(str(sent))
                    else:
                        f.write('\n'+str(sent))