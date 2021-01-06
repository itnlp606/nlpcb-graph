import torch
import pandas as pd
import transformers
from tqdm import tqdm
from model.bert import BERTCLAS, BERT
from utils.utils import print_execute_time
from reader.reader import clas_tensorize
from utils.utils import divide_dataset

@print_execute_time
def clas_predict(args, tokenizer, array, device):
    # load model
    model = torch.load('sent_clas_model/MOD1_5_34_6833', map_location=device)
    
    # load data
    train_data, valid_data = divide_dataset(array, args.num_fold, fold=1)
    valid_iter = clas_tensorize(train_data, tokenizer, args, mode='seq')

    with torch.no_grad():
        model.eval()
        pred_logits, pred_labels = [], []
        for idx, batch_data in enumerate(tqdm(valid_iter)):
            batch_data = tuple(i.to(device) for i in batch_data)
            ids, masks, labels = batch_data
            
            _, logits = model(ids, masks, labels)

            pred_logits.append(logits)
            pred_labels.append(labels)

        precision, recall, F1 = model.calculate_F1(pred_logits, pred_labels)

    print(precision, recall, F1)