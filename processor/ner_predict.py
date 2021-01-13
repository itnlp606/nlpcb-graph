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

def ner_predict(args, tokenizer, device, data_folder):
    vote_knife = 6

    base_dir = 'ner_models'
    mods = os.listdir(base_dir)

    tasks = os.listdir(data_folder)
    for task in tasks:
        # ignore readme
        if task[-3:] == '.md' or task[-4:] == '.git':
            continue

        task_path = 'results/'+task
        if not os.path.exists(task_path):
            os.mkdir(task_path)

        # process
        articles = os.listdir(data_folder+'/'+task)
        pattern = 'Stanza-out.txt'

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
                    model = torch.load(base_dir+'/'+mod).to(device)
                    _, logits = model(ids, masks, labels)
                    logits = torch.argmax(logits, 1)
                    for i in range(len(logits)):
                        vote_box[i] += logits[i].item()

                for idx, vote in enumerate(vote_box):
                    if vote >= vote_knife:
                        result.append(q*args.batch_size + idx + 1)

            # 写进文件
            with open(path+'/entities.txt', 'w') as f:
                for idx, sent in enumerate(result):
                    if idx == 0:
                        f.write(str(sent))
                    else:
                        f.write('\n'+str(sent))