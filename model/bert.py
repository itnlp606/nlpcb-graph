import torch
import torch.nn as nn
from utils.constants import *
from transformers import AutoModelForSequenceClassification

class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
        self.emission = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, \
            cache_dir=args.pretrained_cache_dir, num_labels=2)
        
    def forward(self, ids, masks, labels):
        return self.emission(input_ids=ids, attention_mask=masks, labels=labels)
    
    def calculate_F1(self, pred_logits, pred_labels):
        total_true, total_pred, pred_true = 0, 0, 0
        # for each element in list
        for logits, labels in zip(pred_logits, pred_labels):
            print(logits, labels)
            logits = torch.argmax(logits, 1)
            for logit, label in zip(logits, labels):
                for pred, true in zip(logit, label):
                    pred, true = pred.item(), true.item()
                    if pred == 1: total_pred += 1
                    if true == 1: total_true += 1
                    if pred == ture: pred_true

        try:
            precision = pred_true / total_pred
            recall = pred_true / total_true
            F1 = 2*precision*recall / (precision + recall)
            return precision, recall, F1
        except:
            return 0, 0, 0
