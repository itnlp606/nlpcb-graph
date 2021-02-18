import torch
from tqdm import tqdm
from utils.constants import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def lm_tensorize(data, tokenizer, args, mode='seq'):
    all_tokenized_sents, genes, all_labels = lm_preprocess(data, tokenizer)
    dataset = TensorDataset(all_tokenized_sents['input_ids'], \
        all_tokenized_sents['attention_mask'], genes, all_labels)
    if mode == 'seq':
        sampler = SequentialSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    elif mode == 'random':
        sampler = RandomSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

def lm_preprocess(data, tokenizer):
    sents = [d[0] for d in data]
    labels = [int(d[2]) for d in data]
    genes = torch.tensor([int(d[1]) for d in data])
    tokenized_sents = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
    labels = torch.tensor(labels)
    return tokenized_sents, genes, labels

def relation_tensorize(data, tokenizer, args, mode='seq'):
    all_tokenized_sents, all_labels = relation_preprocess(data, tokenizer)
    dataset = TensorDataset(all_tokenized_sents['input_ids'], \
        all_tokenized_sents['attention_mask'], all_labels)
    if mode == 'seq':
        sampler = SequentialSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=128)

    elif mode == 'random':
        sampler = RandomSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

def relation_preprocess(data, tokenizer):
    sents = [d[0] for d in data]
    labels = [int(d[1]) for d in data]
    tokenized_sents = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
    labels = torch.tensor(labels)
    return tokenized_sents, labels

def ner_tensorize(data, tokenizer, args, mode='seq'):
    # divide into tags and texts
    all_tokenized_sents, all_labels = ner_preprocess(data, tokenizer)
    dataset = TensorDataset(all_tokenized_sents['input_ids'], all_tokenized_sents['attention_mask'],\
        all_tokenized_sents['offset_mapping'], all_labels)
    if mode == 'seq':
        sampler = SequentialSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    elif mode == 'random':
        sampler = RandomSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

# return tokenized_data, labels
def ner_preprocess(data, tokenizer):
    sents, labs = [], []
    for sent, lab in data:
        sents.append(sent)
        char_label = [NER_LABEL2ID['O'] for _ in range(len(sent))]
        for tup in lab:
            start, end, word = tup
            # if sent[start:end] != word:
            #     raise Exception('wrong match')
            char_label[start] = NER_LABEL2ID['B']
            for i in range(start+1, end):
                char_label[i] = NER_LABEL2ID['I']
        labs.append((char_label, lab))

    all_sents, all_labels = [], []
    for sent, (char_label, lab) in zip(sents, labs):
        # print(sent, char_label, lab)
        tokenized_sent = tokenizer(sent, return_offsets_mapping=True)
        label = []
        for idx, mp in enumerate(tokenized_sent['offset_mapping']):
            if idx > 0 and mp[0] == 0 and mp[1] == 0:
                break
            elif idx == 0: 
                label.append(0)
            else:
                ret = char_label[mp[0]]
                if ret == 1:
                    if mp[0] == 0: label.append(ret)
                    elif mp[0] > 0:
                        if char_label[mp[0]-1] == 1:
                            label.append(NER_LABEL2ID['I'])
                        else:
                            label.append(ret)
                else: label.append(ret)

        all_sents.append(sent)
        all_labels.append(label)

    all_tokenized_sents = tokenizer(all_sents, padding=True, truncation=True,\
        return_offsets_mapping=True, return_tensors='pt')
    all_seq_len = all_tokenized_sents['input_ids'].shape[1]

    for label in all_labels:
        label.extend([0]*(all_seq_len - len(label)))

    return all_tokenized_sents, torch.tensor(all_labels)

def clas_tensorize(data, tokenizer, args, mode='seq'):
    # divide into tags and texts
    tokenized_data, labels = clas_preprocess(data, tokenizer)
    if mode == 'seq':
        ids, masks = tokenized_data['input_ids'], tokenized_data['attention_mask']
        dataset = TensorDataset(ids, masks, labels)
        sampler = SequentialSampler(dataset)
        return DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
    elif mode == 'random':
        pos_ids, pos_masks, neg_ids, neg_masks = [], [], [], []
        for i, (ids, mask, label) in enumerate(zip(tokenized_data['input_ids'], \
            tokenized_data['attention_mask'], labels)):
            if label >= 1:
                pos_ids.append(ids)
                pos_masks.append(mask)
            else:
                neg_ids.append(ids)
                neg_masks.append(mask)
        
        pos_ids, pos_masks, neg_ids, neg_masks = torch.stack(pos_ids), \
            torch.stack(pos_masks), torch.stack(neg_ids), torch.stack(neg_masks)
        
        # construct data loader
        pos_dataset = TensorDataset(pos_ids, pos_masks, torch.ones(pos_ids.shape[0], dtype=torch.int64))
        neg_dataset = TensorDataset(neg_ids, neg_masks, torch.zeros(neg_ids.shape[0], dtype=torch.int64))
        pos_sampler, neg_sampler = RandomSampler(pos_dataset), RandomSampler(neg_dataset)
        pos_loader = DataLoader(pos_dataset, sampler=pos_sampler, batch_size=3)
        neg_loader = DataLoader(neg_dataset, sampler=neg_sampler, batch_size=5)
        return pos_loader, neg_loader

# return tokenizer, labels
def clas_preprocess(data, tokenizer):
    labels, texts = [], []
    for tup in data:
        text, label = tup
        labels.append(int(label))
        texts.append(text)
    tokenized_text = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    labels = torch.tensor(labels)
    
    return tokenized_text, labels
