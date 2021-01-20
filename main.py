import torch
import pickle
import pandas as pd
from utils.args import get_parser
from transformers import AutoTokenizer
from processor.clas_predict import clas_predict
from processor.ner_predict import ner_predict
from processor.relation_predict import relation_predict
from processor.ner_train import ner_train
from processor.clas_train import clas_train
from processor.relation_train import relation_train
from processor.lm_train import lm_train
from processor.preprocessor import clas_tensorize, ner_tensorize
from reader.reader import data2numpy

if __name__ == '__main__':
    args = get_parser()

    # basic data structure
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, \
        cache_dir=args.pretrained_cache_dir, use_fast=True)

    # read data
    # with open('array.pkl', 'rb') as f:
    #     array = pickle.load(f)
    clas_array, ner_array, relation_array, type_sent_array, \
        type_ner_array, type_relation_array = data2numpy(args.seed)

    # GPU device
    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device('cuda:'+str(args.gpu_id))
        name = torch.cuda.get_device_name(0)
    else:
        device = torch.device('cpu')
        name = 'cpu'
    
    print("Running on", name)

    if args.task == 'clas':
        array = type_sent_array
        train = clas_train
        predict = clas_predict
    elif args.task == 'ner':
        array = type_ner_array
        train = ner_train
        predict = ner_predict
    elif args.task == 'relation':
        array = type_relation_array
        train = relation_train
        predict = relation_predict
    elif args.task == 'lm':
        array = lm_array
        train = lm_train

    if args.do_train:
        train(args, tokenizer, array, device)
    
    else:
        predict(args, tokenizer, device, 'evaluation-phase1')