import torch
import torch.nn as nn
from utils.constants import *
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

class BERTNER(nn.Module):
    def __init__(self, args, device):
        super(BERTNER, self).__init__()
        self.device = device
        self.emission = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, \
            cache_dir=args.pretrained_cache_dir, num_labels=len(NER_ID2LABEL), return_dict=False)
        
    def forward(self, *args):
        if len(args) == 2:
            pos_data, neg_data = args[0], args[1]
            pos_data = tuple(i.to(self.device) for i in pos_data)
            neg_data = tuple(i.to(self.device) for i in neg_data)
            pos_ids, pos_masks, pos_maps, pos_labels = pos_data
            neg_ids, neg_masks, neg_maps, neg_labels = neg_data

            # concat
            ids = torch.cat((pos_ids, neg_ids), dim=0)
            masks = torch.cat((pos_masks, neg_masks), dim=0)
            labels = torch.cat((pos_labels, neg_labels), dim=0)
            maps = torch.cat((pos_maps, neg_maps), dim=0)

            return self.emission(input_ids=ids, attention_mask=masks, labels=labels)

        elif len(args) == 1:
            batch_data = args[0]
            batch_data = tuple(i.to(self.device) for i in batch_data)
            ids, masks, maps, labels = batch_data
            return self.emission(input_ids=ids, attention_mask=masks, labels=labels)

    def calculate_F1(self, pred_logits, pred_labels):
        total_true, total_pred, pred_true = 0, 0, 0
        # for each element in list
        for logits, labels in zip(pred_logits, pred_labels):
            print(logits.shape, labels.shape)
        
            raise Exception

class BERTCLAS(nn.Module):
    def __init__(self, args, device):
        super(BERTCLAS, self).__init__()
        self.device = device
        self.emission = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, \
            cache_dir=args.pretrained_cache_dir, num_labels=2)
        
    def forward(self, *args):
        if len(args) == 2:
            pos_data, neg_data = args[0], args[1]
            pos_data = tuple(i.to(self.device) for i in pos_data)
            neg_data = tuple(i.to(self.device) for i in neg_data)
            pos_ids, pos_masks, pos_labels = pos_data
            neg_ids, neg_masks, neg_labels = neg_data

            # concat
            ids = torch.cat((pos_ids, neg_ids), dim=0)
            masks = torch.cat((pos_masks, neg_masks), dim=0)
            labels = torch.cat((pos_labels, neg_labels), dim=0)

            return self.emission(input_ids=ids, attention_mask=masks, labels=labels)

        elif len(args) == 1:
            batch_data = args[0]
            batch_data = tuple(i.to(self.device) for i in batch_data)
            ids, masks, labels = batch_data
            return self.emission(input_ids=ids, attention_mask=masks, labels=labels)
    
    def calculate_F1(self, valid_iter, valid_model):
        valid_losses = 0
        pred_logits, pred_labels = [], []
        for idx, batch_data in enumerate(valid_iter):
            batch_data = tuple(i.to(self.device) for i in batch_data)
            ids, masks, labels = batch_data
            print(ids.shape, masks.shape, labels.shape)
            t = self.emission(ids, masks, labels.unsqueeze(1))
            print(t)
            raise Exception

            pred_logits.append(logits)
            pred_labels.append(labels)

            # process loss
            valid_losses += loss.item()
        valid_losses /= len(valid_loader)

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
            return precision, recall, F1, valid_losses
        except:
            return 0, 0, 0, valid_losses