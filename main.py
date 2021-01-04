import torch
import pickle
import pandas as pd
from utils.args import get_parser
from processor.predict import predict
from transformers import AutoTokenizer
from processor.ner_train import ner_train
from processor.clas_train import clas_train
from reader.reader import data2numpy, clas_tensorize, ner_tensorize

if __name__ == '__main__':
    args = get_parser()

    # basic data structure
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, \
        cache_dir=args.pretrained_cache_dir, use_fast=True)

    # read data
    # with open('array.pkl', 'rb') as f:
    #     array = pickle.load(f)
    clas_array, ner_array = data2numpy()

    # GPU device
    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device('cuda:'+str(args.gpu_id))
        name = torch.cuda.get_device_name(0)
    else:
        device = torch.device('cpu')
        name = 'cpu'
    
    print("Running on", name)

    if args.task == 'clas':
        array = clas_array
        train = clas_train
    else:
        array = ner_array
        train = ner_train

    if args.do_train:
        train(args, tokenizer, array, device)
    
    else:
        predict(args, tokenizer, array, device)