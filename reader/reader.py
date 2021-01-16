import os
import json
import pickle
import numpy as np
from copy import deepcopy
from itertools import permutations
from utils.constants import *
from transformers import AutoTokenizer
from collections import defaultdict
from utils.utils import print_execute_time

@print_execute_time
def data2numpy(seed):
    # np.random.seed(seed)
    tasks = os.listdir('data')
    clas_array = []
    ner_array = []
    relation_array = []
    dd = {0:0, 1:0, 2:0}
    tt_num, tt_items = 0, 0
    ct_relations = 0
    ct_code, ct_res = 0, 0
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', \
    #     cache_dir='pretrained_models', use_fast=True)

    for task in tasks:
        # if task != 'natural_language_inference':
        #     continue

        # ignore readme
        if task[-3:] == '.md' or task[-4:] == '.git':
            continue

        # process
        articles = os.listdir('data/'+task)
        pattern = 'Stanza-out.txt'
        paragraph_pat = 'Grobid-out.txt'
        sent_pat = 'sentences.txt'
        entity_pat = 'entities.txt'

        for article in articles:
            tt_num += 1
            # 提取句子文件
            files = os.listdir('data/'+task+'/'+article)
            for f in files:
                if pattern in f:
                    name = f
                if paragraph_pat in f:
                    para_name = f
            
            # get dir, data
            base_article_dir = 'data/'+task+'/'+article+'/'
            para_dir = 'data/'+task+'/'+article+'/'+para_name
            sent_dir = 'data/'+task+'/'+article+'/'+name
            label_dir = 'data/'+task+'/'+article+'/'+sent_pat
            entity_dir = 'data/'+task+'/'+article+'/'+entity_pat

            with open(para_dir, 'r') as f:
                paras = f.readlines()
            with open(sent_dir, 'r') as f:
                sents = f.readlines()
            with open(label_dir, 'r') as f:
                labels = f.readlines()
            with open(entity_dir, 'r') as f:
                entities = f.readlines()
            labels = [int(label) for label in labels]
            
            def get_context(i):
                if i < 0 or i >= len(sents):
                    return ""
                return '#' + sents[i]

            # process entities
            sentID2entites = defaultdict(list)
            for entity in entities:
                if entity[-1] == '\n': entity = entity[:-1]
                items = entity.split('\t')
                
                items[0], items[1], items[2] = \
                    int(items[0]), int(items[1]), int(items[2])
                
                # 实体标注还是很准的，就两个多空格
                if items[3] != sents[items[0]-1][items[1]:items[2]]:
                    items[3] = items[3].strip()

                sentID2entites[int(items[0])].append((\
                    items[1], items[2], items[3]))

            # 提取三元组，要和句子结合，提取每个句子的三元组。
            triple_files = os.listdir(base_article_dir + 'triples')
            for name in triple_files:
                with open(base_article_dir+'triples/'+name, 'r') as f:
                    content = f.readlines()
                    triples = set()
                    for line in content:
                        if line[-1] == '\n': line = line[:-1]
                        # 替换setup
                        if 'Experimental Setup' in line:
                            line = line.replace('Experimental Setup', 'Experimental setup')
                        
                        # 去括号
                        ents = line[1:-1].split('||')
                        triples.add(tuple(ents))
                        tt_items += 1

                    # 对句子中的实体排序（插入排序）
                    for sent in sentID2entites:
                        entity_list = sentID2entites[sent]

                        sorted_entities = [entity_list[0]]
                        for i in range(1, len(entity_list)):
                            entity = entity_list[i]
                            for j, ele in enumerate(sorted_entities):
                                if ele[0] > entity[0]:
                                    sorted_entities.insert(j, entity)
                                    break
                            else:
                                sorted_entities.append(entity)

                        sorted_entities = [i[2] for i in sorted_entities]

                        # 对每个三元组，如果在，看位置
                        pos_pos = []
                        for triple in triples:
                            # code, research情况
                            if FILENAME2BLOCK[name] in BLOCK_MID_NAMES \
                                and triple[-1] in sorted_entities:
                                pos_sample = deepcopy(sents[sent-1])
                                for word in triple:
                                    pos_sample += '#' + word
                                relation_array.append((pos_sample, BLOCK2ID[FILENAME2BLOCK[name]]))
                                pos_pos.append([sorted_entities.index(triple[-1])])
                                continue

                            # 头结点的has和其他谓词情况
                            if triple[0] in BLOCK_BACK_NAMES:
                                # 基本都是不存在的谓词，先忽略此情况
                                pass

                            for word in triple:
                                if word not in sorted_entities:
                                    break
                            else:
                                # 存在三元组，生成正样本(三元组加到后面)
                                pos_sample = deepcopy(sents[sent-1])
                                for word in triple:
                                    pos_sample += '#' + word

                                relation_array.append((pos_sample, BLOCK2ID[FILENAME2BLOCK[name]]))
                                pos = [sorted_entities.index(word) for word in triple]
                                pos_pos.append(pos)

                        # 生成负样本
                        lp = len(pos_pos)
                        ls = len(sorted_entities)
                        neg_beishu = 3
                        if pos_pos:
                            neg_pos = []
                            array = list(range(ls))
                            if FILENAME2BLOCK[name] in BLOCK_MID_NAMES:
                                while len(neg_pos) < min(neg_beishu*lp, ls-lp):
                                    pos = np.random.choice(array, 1).tolist()
                                    if pos in pos_pos or pos in neg_pos:
                                        continue
                                    neg_pos.append(pos)
                                for pos in neg_pos:
                                    ents = [sorted_entities[p] for p in pos]
                                    neg_sample = deepcopy(sents[sent-1])
                                    for word in ents:
                                        neg_sample += '#' + word
                                    relation_array.append((neg_sample, 0))
                                continue

                            # 颠倒顺序 2个
                            # for p in pos_pos:
                            #     perms = list(permutations(p))
                            #     ords = np.random.permutation(6) # 3元组，6个顺序
                            #     ct = 0
                            #     rt = 0
                            #     while len(neg_pos) < min(neg_beishu*lp, ls*(ls-1)*(ls-2)-lp) and rt < 2:
                            #         ele = list(perms[ords[ct]])
                            #         if ele not in pos_pos and ele not in neg_pos:
                            #             neg_pos.append(ele)
                            #             rt += 1
                            #         ct += 1

                            # # 随机替换一个词
                            # if ls > 3:
                            #     for pos in pos_pos:
                            #         rt = 0
                            #         ct = 0
                            #         while len(neg_pos) < min(neg_beishu*lp, ls*(ls-1)*(ls-2)-lp) and rt < 2:
                            #             # 随机选取一个位置
                            #             p = np.random.randint(3)

                            #             # 随机选取一个实体
                            #             e = np.random.randint(len(sorted_entities))
                            #             while e in pos:
                            #                 e = np.random.randint(len(sorted_entities))
                            #                 # print(e, pos)

                            #             can = deepcopy(pos)
                            #             can[p] = e

                            #             if can not in pos_pos and can not in neg_pos:
                            #                 neg_pos.append(can)
                            #                 rt += 1
                            #             ct += 1

                            # 随机选择
                            while len(neg_pos) < min(neg_beishu*lp, ls*(ls-1)*(ls-2)-lp):
                                pos = np.random.choice(array, 3, replace=False).tolist()
                                if pos in pos_pos or pos in neg_pos:
                                    continue
                                neg_pos.append(pos)

                            for pos in neg_pos:
                                ents = [sorted_entities[p] for p in pos]
                                neg_sample = deepcopy(sents[sent-1])
                                for word in ents:
                                    neg_sample += '#' + word
                                relation_array.append((neg_sample, 0))
                                ct_relations += 1
                                if ct_relations % 350 == 0:
                                    ct_code += 3
                                    for word in ents:
                                        code_neg = deepcopy(sents[sent-1])
                                        code_neg += '#Contribution#Code#' + word
                                        relation_array.append((code_neg, 0))
                                if ct_relations % 20 == 0:
                                    ct_res += 3
                                    for word in ents:
                                        res_neg = deepcopy(sents[sent-1])
                                        res_neg += '#Contribution#has research problem#' + word
                                        relation_array.append((res_neg, 0))

            # append data
            for i, sent in enumerate(sents):
                # 去掉回车
                if sent[-1] == '\n':
                    sent = sent[:-1]

                # 加特征前处理NER问题
                if i+1 in labels:
                    ner_array.append((sent, sentID2entites[i+1]))

                # add feature
                # extract title
                for j, para in enumerate(paras):
                    if sent in para:
                        idx = j

                while idx >= 0 and paras[idx] != '\n': idx -= 1
                if idx == 0: title = paras[0]
                else: title = paras[idx+1]

                if title[-1] == '\n':
                    title = title[:-1]
                
                sent += '#' + title

                sent += get_context(i-1) + get_context(i+1)

                # s = tokenizer(sent)
                # if len(s['input_ids']) < 510:
                #     dd[0] += 1
                # else: 
                #     dd[1] += 1
                #     if i+1 in labels:
                #         dd[2] += 1

                if i+1 in labels:
                    clas_array.append((sent, 1))#label_id))
                else:
                    clas_array.append((sent, 0))

    # with open('array.pkl', 'wb') as f:
    #     pickle.dump(np.array(array), f)
    # print(tt_num, tt_items, tt_items/tt_num)
    # raise Exception

    return np.array(clas_array), np.array(ner_array, dtype=object), np.array(relation_array)