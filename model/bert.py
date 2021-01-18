import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from utils.constants import *
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, \
    AutoModel

class BERTLM(nn.Module):
    def __init__(self, args):
        super(BERTLM, self).__init__()
        self.emission = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, \
            cache_dir=args.pretrained_cache_dir, num_labels=2)
        
    def forward(self, ids, masks, labels):
        return self.emission(input_ids=ids, attention_mask=masks, labels=labels).to_tuple()

    def calculate_F1(self, pred_logits, pred_labels, gs):
        id2oktup = {}

        totals, trues = 0, 0
        # for each element in list
        for logits, labels, genes in zip(pred_logits, pred_labels, gs):
            for pred, true, gene in zip(logits, labels, genes):
                pred, true, gene = pred[1].item(), true.item(), gene.item()
                if gene not in id2oktup:
                    id2oktup[gene] = (pred, true)
                elif pred > id2oktup[gene][0]:
                    id2oktup[gene] = (pred, true)

        # 提取pred_true啥的
        for id0 in id2oktup:
            pred = id2oktup[id0]
            if pred[1] == 1:
                trues += 1
            totals += 1

        return trues/totals

class BERTRE(nn.Module):
    def __init__(self, args):
        super(BERTRE, self).__init__()
        self.emission = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, \
            cache_dir=args.pretrained_cache_dir, num_labels=len(ID2BLOCK)+1)
        
    def forward(self, ids, masks, labels):
        return self.emission(input_ids=ids, attention_mask=masks, labels=labels).to_tuple()

    def calculate_F1(self, pred_logits, pred_labels):
        zeros = [0 for _ in range(len(ID2BLOCK))]
        total_true, total_pred, pred_true = \
            deepcopy(zeros), deepcopy(zeros), deepcopy(zeros)
        # for each element in list
        for logits, labels in zip(pred_logits, pred_labels):
            logits = torch.argmax(logits, 1)
            for pred, true in zip(logits, labels):
                pred, true = pred.item(), true.item()
                if pred >= 1 and true == pred:
                    pred_true[pred-1] += 1
                if pred >= 1:
                    total_pred[pred-1] += 1
                if true >= 1:
                    total_true[true-1] += 1

        F1s, pres, recalls = deepcopy(zeros), deepcopy(zeros), deepcopy(zeros)
        for i in range(len(F1s)):
            try:
                pres[i] = pred_true[i] / total_pred[i]
                recalls[i] = pred_true[i] / total_true[i]
                F1s[i] = 2*pres[i]*recalls[i] / (pres[i] + recalls[i])
            except:
                pres[i], recalls[i], F1s[i] = 0, 0, 0

        return np.mean(pres), np.mean(recalls), np.mean(F1s)

class BERTNER(nn.Module):
    def __init__(self, args):
        self.args = args
        super(BERTNER, self).__init__()
        self.emission = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, \
            cache_dir=args.pretrained_cache_dir, num_labels=len(NER_ID2LABEL))
        if self.args.use_crf:
            from allennlp.modules import ConditionalRandomField
            self.crf = ConditionalRandomField(len(NER_ID2LABEL), include_start_end_transitions=False)
        
    def forward(self, ids, masks, labels):
        loss, logits = self.emission(input_ids=ids, attention_mask=masks, labels=labels).to_tuple()

        if self.args.use_crf:        
            loss = -self.crf(logits, labels, masks)
            logits = self.crf.viterbi_tags(logits, masks)
            return loss, logits

        logits = torch.argmax(logits, 2)
        return loss, logits
 
    def calculate_F1(self, pred_logits, pred_labels):
        total_true, total_pred, pred_true = 0, 0, 0
        # for each element in list
        for logits, labels in zip(pred_logits, pred_labels):
            # argmax后会降维

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

class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
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