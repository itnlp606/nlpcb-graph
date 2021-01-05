import torch
import torch.nn as nn
from tqdm import tqdm
from torchcrf import CRF
from utils.constants import *
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

class BERTNER(nn.Module):
    def __init__(self, args):
        super(BERTNER, self).__init__()
        self.emission = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, \
            cache_dir=args.pretrained_cache_dir, num_labels=len(NER_ID2LABEL))
        self.crf = CRF(len(NER_ID2LABEL))
        
    def forward(self, ids, masks, labels):
        _, logits = self.emission(input_ids=ids, attention_mask=masks, labels=labels).to_tuple()
        print(logits.shape, labels.dtype, masks.dtype)
        loss = -self.crf(logits, labels, masks)
        logits = self.crf.decode(logits)
        return loss, logits
 
    def calculate_F1(self, pred_logits, pred_labels):
        total_true, total_pred, pred_true = 0, 0, 0
        # for each element in list
        for logits, labels in zip(pred_logits, pred_labels):
            # argmax后会降维
            logits = torch.argmax(logits, 2)

            # logits, labels shape: (16, 414)
            for pred, true in zip(logits, labels):
                trues, preds = [], []
                # 看true预测的实体， 每个i是1*1 tensor
                i = 0
                while i < len(true):
                    j = i+1
                    if true[i] == 1:
                        while j < len(true) and true[j] == 2:
                            j += 1
                        trues.append((i, j))
                    i = j
                # pred预测的实体
                i = 0
                while i < len(pred):
                    j = i+1
                    if pred[i] == 1:
                        while j < len(pred) and pred[j] == 2:
                            j += 1
                        preds.append((i, j))
                    i = j

                for pd in preds:
                    if pd in trues:
                        pred_true += 1
                total_true += len(trues)
                total_pred += len(preds)

        try:
            precision = pred_true / total_pred
            recall = pred_true / total_true
            F1 = 2*precision*recall / (precision + recall)
            return precision, recall, F1
        except:
            return 0, 0, 0

class BERTCLAS(nn.Module):
    def __init__(self, args):
        super(BERTCLAS, self).__init__()
        self.emission = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, \
            cache_dir=args.pretrained_cache_dir, num_labels=2)
        
    def forward(self, ids, masks, labels):
        return self.emission(input_ids=ids, attention_mask=masks, labels=labels).to_tuple()
    
    def calculate_F1(self, pred_logits, pred_labels):
        total_true, total_pred, pred_true = 0, 0, 0
        # for each element in list
        for logits, labels in zip(pred_logits, pred_labels):
            logits = torch.argmax(logits, 1)
            for pred, true in zip(logits, labels):
                pred, true = pred.item(), true.item()
                if pred >= 1 and true >= 1:
                    pred_true += 1
                if pred >= 1:
                    total_pred += 1
                if true >= 1:
                    total_true += 1

        try:
            precision = pred_true / total_pred
            recall = pred_true / total_true
            F1 = 2*precision*recall / (precision + recall)
            return precision, recall, F1
        except:
            return 0, 0, 0