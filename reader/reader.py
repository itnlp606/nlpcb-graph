import os
import json
import pickle
import numpy as np
from copy import deepcopy
from itertools import permutations
from utils.utils import C
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
    type_sent_array = []
    type_ner_array = []
    type_relation_array = []
    lm_sents = []
    lm_id = 1

    # 临时变量区
    contris, type2, trip3, typeht, tupht, cr = 0, 0, 0, 0, 0, 0
    ccr, rrr = 0, 0
    ddd, ppp = 0, 0
    len_t = 0
    ss = set()
    
    ct_relations = 0
    ct_code, ct_res = 0, 0
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', \
    #     cache_dir='pretrained_models', use_fast=True)

    for task in tasks:
        # 不要再取消注释了
        # if task == 'natural_language_inference':
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
        type_tmp_sent = []
        type_tmp_ner = []
        type_tmp_relation = []

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
            tt_items += len(entities)
            
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
                    len_t += len(triples)

                    yu_triples = deepcopy(triples)

                    for sent in sentID2entites:
                        # 对句子中的实体排序（插入排序）
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
                        has_pos = []
                        for triple in triples:
                            # code, research情况
                            if FILENAME2BLOCK[name] in BLOCK_MID_NAMES \
                                and triple[-1] in sorted_entities:
                                pos_sample = deepcopy(sents[sent-1])
                                for word in triple:
                                    pos_sample += '#' + word
                                relation_array.append((pos_sample, BLOCK2ID[FILENAME2BLOCK[name]]))
                                type_tmp_relation.append((pos_sample, BLOCK2ID[FILENAME2BLOCK[name]]))
                                cr += 1
                                pos_pos.append([sorted_entities.index(triple[-1])])
                                yu_triples.discard(triple)
                                continue

                            # 除了代码，研究，去掉Contribution
                            if triple[0] == 'Contribution':
                                if triple in yu_triples:
                                    yu_triples.discard(triple)
                                    contris += 1
                                continue

                            # 头部是类别，后两个是实体
                            if triple[0] in BLOCK2FILENAME and triple[1] in sorted_entities \
                                and triple[2] in sorted_entities:
                                pp1, pp2 = sorted_entities.index(triple[1]), \
                                     sorted_entities.index(triple[2])
                                if pp1 < pp2:
                                    # 加入正样本
                                    pos_pos.append([pp1, pp2])
                                    pos_sample = deepcopy(sents[sent-1])
                                    pos_sample += '#type' + '#' + triple[1] + '#' + triple[2]
                                    relation_array.append((pos_sample, BLOCK2ID[FILENAME2BLOCK[name]]))
                                    type_tmp_relation.append((pos_sample, BLOCK2ID[FILENAME2BLOCK[name]]))

                                type2 += 1
                                yu_triples.discard(triple)
                                continue

                            for kd, word in enumerate(triple):
                                if word not in sorted_entities:
                                    break
                                    # 若中途退出，说明三元组不在句子里
                            else:
                                # 存在三元组，生成正样本(三元组加到后面)
                                pos_sample = deepcopy(sents[sent-1])
                                for word in triple:
                                    pos_sample += '#' + word

                                relation_array.append((pos_sample, BLOCK2ID[FILENAME2BLOCK[name]]))
                                type_tmp_relation.append((pos_sample, BLOCK2ID[FILENAME2BLOCK[name]]))
                                pos = [sorted_entities.index(word) for word in triple]
                                if pos[0] < pos[1] < pos[2]:
                                    pos_pos.append(pos)
                                trip3 += 1
                                yu_triples.discard(triple)

                            # 类别元组+has+普通元祖
                            if triple[0] in BLOCK2FILENAME and triple[1] == 'has' and triple[2] in sorted_entities:
                                typeht += 1
                                ps = sorted_entities.index(triple[2])
                                pos_pos.append([ps])
                                pos_sample = deepcopy(sents[sent-1])
                                pos_sample += '#type#has' + '#' + triple[2]
                                relation_array.append((pos_sample, BLOCK2ID[FILENAME2BLOCK[name]]))
                                type_tmp_relation.append((pos_sample, BLOCK2ID[FILENAME2BLOCK[name]]))
                                yu_triples.discard(triple)
                            
                            # 普通元组+has+普通元祖
                            if triple[1] == 'has' and triple in yu_triples \
                                and triple[0] in sorted_entities and triple[2] in sorted_entities:
                                tupht += 1
                                pp1, pp2 = sorted_entities.index(triple[0]), \
                                     sorted_entities.index(triple[2])
                                has_pos.append([pp1, pp2])

                                yu_triples.discard(triple)
                                pos_sample = deepcopy(sents[sent-1])
                                pos_sample += '#' + triple[0] + '#has#' + triple[2]
                                relation_array.append((pos_sample, BLOCK2ID[FILENAME2BLOCK[name]]))
                                type_tmp_relation.append((pos_sample, BLOCK2ID[FILENAME2BLOCK[name]]))

                        # 生成负样本
                        lp3, lp2, lp1 = 0, 0, 0
                        ls = len(sorted_entities)
                        neg_beishu = 4
                        array = list(range(ls))

                        # 单独处理has_pos
                        neg_has = []
                        len_has = len(has_pos)
                        for pos in has_pos:
                            len_h = min(neg_beishu*len_has, int(C(ls, 2))-len_has)
                            # 替换第一个实体
                            ps = list(np.random.choice(array[0:pos[1]], pos[1], replace=False))
                            for p in ps:
                                tmp_pos = [p, pos[1]]
                                if tmp_pos not in has_pos and tmp_pos not in neg_has:
                                    neg_has.append(tmp_pos)
                                    break
                            
                            # 替换第二个实体
                            num = ls-pos[0]-1
                            ps = list(np.random.choice(array[pos[0]+1:], num, replace=False))
                            for p in ps:
                                tmp_pos = [pos[0], p]
                                if tmp_pos not in has_pos and tmp_pos not in neg_has:
                                    neg_has.append(tmp_pos)
                                    break        
                                
                            # 随机选择
                            while len(neg_has) < len_h:
                                pos = np.random.choice(array, 2, replace=False).tolist()
                                pos.sort()
                                if pos in has_pos or pos in neg_has:
                                    continue
                                neg_has.append(pos)                               
                                
                            # 将neg_has里的元组变成句子放入数组中
                            for pos in neg_has:
                                ents = [sorted_entities[p] for p in pos]
                                neg_sample = deepcopy(sents[sent-1])
                                neg_sample += '#' + ents[0] + '#has'
                                neg_sample += '#' + ents[1]
                                relation_array.append((neg_sample, 0))
                                type_tmp_relation.append((neg_sample, 0))

                        if pos_pos and FILENAME2BLOCK[name] not in BLOCK_MID_NAMES:
                            neg_pos = []

                            for pp in pos_pos:
                                if len(pp) == 3: lp3 += 1
                                elif len(pp) == 2: lp2 += 1
                                elif len(pp) == 1: lp1 += 1

                            len1 = min(neg_beishu*lp1, ls-lp1)
                            len2 = min(neg_beishu*lp2, int(C(ls, 2))-lp2)
                            len3 = min(neg_beishu*lp3, int(C(ls, 3))-lp3)

                            # 随机替换一个词
                            array = [i for i in range(len(sorted_entities))]
                            neg_1, neg_2, neg_3 = [], [], []
                            for udx, pos in enumerate(pos_pos):
                                if len(pos) == 2:
                                    # 替换第一个实体
                                    ps = list(np.random.choice(array[0:pos[1]], pos[1], replace=False))
                                    for p in ps:
                                        tmp_pos = [p, pos[1]]
                                        if tmp_pos not in pos_pos and tmp_pos not in neg_2:
                                            neg_2.append(tmp_pos)
                                            break
                                    
                                    # 替换第二个实体
                                    num = ls-pos[0]-1
                                    ps = list(np.random.choice(array[pos[0]+1:], num, replace=False))
                                    for p in ps:
                                        tmp_pos = [pos[0], p]
                                        if tmp_pos not in pos_pos and tmp_pos not in neg_2:
                                            neg_2.append(tmp_pos)
                                            break                     

                                elif len(pos) == 3:
                                    # 替换第一个实体
                                    ps = list(np.random.choice(array[0:pos[1]], pos[1], replace=False))
                                    for p in ps:
                                        tmp_pos = [p, pos[1], pos[2]]
                                        if tmp_pos not in pos_pos and tmp_pos not in neg_3:
                                            neg_3.append(tmp_pos)
                                            break
                                    
                                    # 替换第二个实体
                                    num = pos[2]-pos[0]-1
                                    ps = list(np.random.choice(array[pos[0]+1:pos[2]], num, replace=False))
                                    for p in ps:
                                        tmp_pos = [pos[0], p, pos[2]]
                                        if tmp_pos not in pos_pos and tmp_pos not in neg_3:
                                            neg_3.append(tmp_pos)
                                            break

                                    # 替换第三个实体
                                    num = ls-pos[1]-1
                                    ps = list(np.random.choice(array[pos[1]+1:], num, replace=False))
                                    for p in ps:
                                        tmp_pos = [pos[0], pos[1], p]
                                        if tmp_pos not in pos_pos and tmp_pos not in neg_3:
                                            neg_3.append(tmp_pos)
                                            break

                            # 随机选择：对2实体元组 
                            while len(neg_1) < len1:
                                pos = np.random.choice(array, 1, replace=False).tolist()
                                if pos in pos_pos or pos in neg_1:
                                    continue
                                neg_1.append(pos)

                            # 随机选择：对2实体元组
                            while len(neg_2) < len2:
                                pos = np.random.choice(array, 2, replace=False).tolist()
                                pos.sort()
                                if pos in pos_pos or pos in neg_2:
                                    continue
                                neg_2.append(pos)

                            # 随机选择：对3
                            while len(neg_3) < len3:
                                pos = np.random.choice(array, 3, replace=False).tolist()
                                pos.sort()
                                if pos in pos_pos or pos in neg_3:
                                    continue
                                neg_3.append(pos)

                            neg_pos = neg_1 + neg_2 + neg_3

                            # 将neg_pos里的元组变成句子放入数组中
                            for pos in neg_pos:
                                ents = [sorted_entities[p] for p in pos]
                                neg_sample = deepcopy(sents[sent-1])

                                if len(pos) == 1:
                                    neg_sample += '#type#has'

                                if len(pos) == 2:
                                    neg_sample += '#type'

                                for word in ents:
                                    neg_sample += '#' + word
                                relation_array.append((neg_sample, 0))
                                type_tmp_relation.append((neg_sample, 0))
                                ct_relations += 1
                                if ct_relations % 350 == 0:
                                    ct_code += 3
                                    for word in ents:
                                        code_neg = deepcopy(sents[sent-1])
                                        code_neg += '#Contribution#Code#' + word
                                        ccr += 1
                                        relation_array.append((code_neg, 0))
                                        type_tmp_relation.append((code_neg, 0))
                                if ct_relations % 20 == 0:
                                    ct_res += 3
                                    for word in ents:
                                        res_neg = deepcopy(sents[sent-1])
                                        res_neg += '#Contribution#has research problem#' + word
                                        rrr += 1
                                        relation_array.append((res_neg, 0))
                                        type_tmp_relation.append((res_neg, 0))

                    
                    # if len(yu_triples) > 0:
                    #     print(yu_triples)
                    # o += len(yu_triples)

            # append data
            for i, sent in enumerate(sents):
                # 去掉回车
                if sent[-1] == '\n':
                    sent = sent[:-1]

                # 加特征前处理NER问题
                if i+1 in labels:
                    ner_array.append((sent, sentID2entites[i+1]))
                    type_tmp_ner.append((sent, sentID2entites[i+1]))

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

                KK = 2
                for cpt in range(1, KK+1):
                    sent += get_context(i-cpt) + get_context(i+cpt)

                # s = tokenizer(sent)
                # if len(s['input_ids']) < 510:
                #     dd[0] += 1
                # else: 
                #     dd[1] += 1
                #     if i+1 in labels:
                #         dd[2] += 1

                if i+1 in labels:
                    clas_array.append((sent, 1))#label_id))
                    type_tmp_sent.append((sent, 1))
                else:
                    clas_array.append((sent, 0))
                    type_tmp_sent.append((sent, 0))

        type_sent_array.append(type_tmp_sent)    
        type_ner_array.append(type_tmp_ner)
        type_relation_array.append(type_tmp_relation)
        # for i in type_tmp_relation:
        #     print(i)

    # with open('array.pkl', 'wb') as f:
    #     pickle.dump(np.array(array), f)
    # print(tt_num, tt_items, tt_items/tt_num)
    # print(contris, type2, trip3, typeht, tupht, cr)

    return np.array(clas_array), np.array(ner_array, dtype=object), np.array(relation_array), \
        type_sent_array, type_ner_array, type_relation_array