import os
import torch
import pandas as pd
import transformers
from tqdm import tqdm
from model.bert import BERTCLAS, BERT
from utils.utils import print_execute_time
from processor.preprocessor import clas_tensorize
from utils.utils import divide_dataset
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

@print_execute_time
def clas_predict(args, tokenizer, device, data_folder):
    vote_knife = 4

    base_dir = 'sent_clas_models'
    mods = os.listdir(base_dir)
    
    models = []
    for mod in mods:
        model = torch.load('sent_clas_models/'+mod)#, map_location=device)
        models.append(model)

    tasks = os.listdir(data_folder)
    for task in tasks:
        # ignore readme
        if task[-3:] == '.md' or task[-4:] == '.git':
            continue

        task_path = 'results/'+task
        if not os.path.exists(task_path):
            os.mkdir(task_path)
        else:
            continue

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
            with open(sent_dir, 'r') as f:
                sents = f.readlines()

            labels = torch.tensor([0]*len(sents))
            tokenized_sents = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
            dataset = TensorDataset(tokenized_sents['input_ids'], tokenized_sents['attention_mask'], labels)
            loader = DataLoader(dataset, args.batch_size)

            for q, data in enumerate(loader):
                data = tuple(i.to(device) for i in data)
                ids, masks, labels = data
                
                vote_box = [0 for _ in range(args.batch_size)]
                for model in models:
                    model = model.to(device)
                    _, logits = model(ids, masks, labels)
                    model = model.to('cpu')
                    logits = torch.argmax(logits, 1)
                    for i in range(len(logits)):
                        vote_box[i] += logits[i].item()
                        
                data = tuple(i.to('cpu') for i in data)

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