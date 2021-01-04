import torch
import pandas as pd
import transformers
from model.bert import BERTCLAS
from utils.utils import print_execute_time
from reader.reader import clas_tensorize
from utils.utils import divide_dataset

@print_execute_time
def predict(args, tokenizer, array, device):
    # load model
    model = torch.load('trained_models/MOD1_9', map_location=device)
    
    # load data
    _, valid_data = divide_dataset(array, args.num_fold, fold=1)
    valid_loader, valid_maps = tensorize(valid_data, tokenizer, args, mode='seq')

    model.eval()
    with torch.no_grad():
        valid_losses = 0
        pred_logits, pred_labels = [], []
        for idx, batch_data in enumerate(valid_loader):
            batch_data = tuple(i.to(device) for i in batch_data)
            ids, masks, labels = batch_data
            loss, logits = model(ids, masks, labels).to_tuple()

            # process loss
            valid_losses += loss.item()

        valid_losses /= len(valid_loader)

    print(valid_losses)