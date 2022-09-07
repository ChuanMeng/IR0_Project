#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:54:12 2022

@author: c.meng
"""
import datetime
import numpy as np
from Data_loader import passage_loader, query_loader
from Preprocessing import process_passage, process_query
from Tfidf import tfidf 
from RankNet import train, inference
import argparse
from Metrics import ndcg,mrr
import codecs
import os
import torch
import collections
import json


def full_ranking(args, passages_tokenised, p_index2id, p_id2len, queries_tokenised, q_index2id):
    
    full_ranker = tfidf(passages_tokenised)
    
    scores = []
    for q in queries_tokenised:
        score = full_ranker.simall(q) # [passage_num]
        scores.append(score)

    scores = np.array(scores) # [query_num, passage_num]
    
    
    # build_features
    top_p_indexs = np.argsort(-scores, 1)[:, :args.topk] # [query_num, 100]
    
    with codecs.open(args.dataset+"_full_ranking.text", "w", "utf-8") as file:
        for query_index, p_scores in enumerate(scores):
            row_num=0
            for top_p_index in top_p_indexs[query_index]:
                row_num+=1
                
                query_id = q_index2id[query_index]
                p_id = p_index2id[top_p_index]
                ranking =  row_num
                
                feature_1 = p_scores[top_p_index]
                feature_2 = p_id2len[p_index2id[top_p_index]]
                
                file.write('\t'.join([query_id, p_id, str(ranking), str(feature_1), str(feature_2), args.ranking+"_"+args.dataset])+os.linesep)
                
    return None



def re_ranking(args):
    
    if args.mode=="train":
        train(args)
    elif args.mode=="inference":
        inference(args) # [query_num, 100]
    else:
        raise Exception
    
    
    

def evaluation(args):
        
    # evaluation
    if args.ranking=="full_ranking":
        # full_ranking
        path = args.dataset+"_full_ranking.text"
        
    elif args.ranking=="re_ranking":
        path = args.dataset+"_re_ranking.text"
    else:
        raise Exception
    
    
    label = json.load(open("data/"+args.dataset+"_labels.json", 'r'))
    
    logger= collections.defaultdict(dict)
    
    with codecs.open(path, "r", "utf-8") as file:
        for line in file.readlines():
            content = line.split('\t')
            
            logger[content[0]][int(content[2])]=content[1]

        
    
    
    print('MRR@{}: {:.4f}'.format(args.topk, mrr(logger, label, args.topk)))
    print('NDCG@{}: {:.4f}'.format(args.topk,ndcg(logger, label, args.topk)))
    


    

    
def main(args):    
    
    
    if args.mode=="evaluation":
        evaluation(args)
        exit()
    
    # data loader
    passages = passage_loader(args.p_path)
    
    if args.dataset=="train":
        query_path = args.train_query_path
    elif args.dataset=="validation":
        query_path = args.validation_query_path
    elif args.dataset=="test":
        query_path = args.test_query_path

    else:
        raise Exception
    
    queries = query_loader(query_path)

    
    
    # preprocessing and indexing
    passages_tokenised, p_id2index, p_index2id, p_id2len = process_passage(passages)
    queries_tokenised, q_index2id = process_query(p_id2index, queries)


    if args.ranking=="full_ranking":
        # full_ranking
        full_ranking(args, passages_tokenised, p_index2id, p_id2len, queries_tokenised, q_index2id)
        
    elif args.ranking=="re_ranking":
        re_ranking(args)
    else:
        raise Exception
    
    
    # evaluation
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='pipeline')
    parser.add_argument("--mode", type=str, default='evaluation') # [train, inference, evaluation]
    parser.add_argument("--dataset", type=str, default='test') # [train, valiation, test]
    parser.add_argument("--ranking", type=str, default='full_ranking') # [full_ranking,re_ranking]
    parser.add_argument("--topk", type=int, default=100) 
    
    parser.add_argument("--p_path", type=str, default='data/passages.json')
    parser.add_argument("--train_query_path", type=str, default='data/train_queries.json')
    parser.add_argument("--validation_query_path", type=str, default='data/validation_queries.json')
    parser.add_argument("--test_query_path", type=str, default='data/test_queries.json')
    
    parser.add_argument("--train_label_path", type=str, default='data/train_labels.json')
    parser.add_argument("--validation_label_path", type=str, default='data/validation_labels.json')
    parser.add_argument("--test_label_path", type=str, default='data/test_labels.json')
    
    
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--input_size", type=float, default=2)
    parser.add_argument("--hidden_size1", type=float, default=8)
    parser.add_argument("--hidden_size2", type=float, default=8)
    parser.add_argument("--output_size", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    
    
    parser.add_argument("--replicability", action='store_true')
    parser.add_argument("--random_seed", type=int, default=0)


    args = parser.parse_args()
    
    if args.replicability:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    main(args)