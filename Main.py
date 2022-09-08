
"""
Created on Tue Aug 30 17:54:12 2022

@author: c.meng
"""
import datetime
import numpy as np
from Data_loader import passage_loader, query_loader, label_loader
from Preprocessing import process_passage, process_query
from Tfidf import tfidf 
from RankNet import train, inference
import argparse
from Metrics import mrr,ndcg
import codecs
import os
import torch
import collections
import json


def full_ranking(args):

    # data loading
    passages = passage_loader(os.path.join(args.data_path, args.passage_file))
    queries = query_loader(os.path.join(args.data_path, args.dataset+"_"+args.query_file))

    # preprocessing and indexing
    passages_tokenised, p_id2len = process_passage(passages)
    queries_tokenised= process_query(queries)
    
    full_ranker = tfidf(passages_tokenised)
    scores = full_ranker.sim(queries_tokenised)
    
    # output the result file & build_features  
    for q_id, p2score in scores.items():
        sorted_p2score=sorted(p2score.items(), key=lambda x:x[1], reverse = True)
        scores[q_id]=sorted_p2score
        
    
    with codecs.open(os.path.join(args.output_path, args.ranking_type+"_"+args.dataset+"_"+args.result_file), "w", "utf-8") as file:
        for q_id, p2score in scores.items():
            row_num=0
            for (p_id, score) in p2score[:args.topk]:
                row_num+=1         
                ranking =  row_num
                feature_1 = score
                feature_2 = p_id2len[p_id]             
    
                file.write('\t'.join([q_id, p_id, str(ranking), str(feature_1), str(feature_2), args.ranking_type+"_"+args.dataset])+os.linesep) 
                
    print("Produce file {}".format(os.path.join(args.output_path, "full_ranking"+"_"+args.dataset+"_"+args.result_file)))        
    return None


def re_ranking(args):
    if args.mode=="train":
        
        q_id = []
        features = []
        labels = []
        
        #print("Load file {}".format(os.path.join(args.data_path, args.dataset+"_"+args.label_file)))
        q2labels = label_loader(os.path.join(args.data_path, args.dataset+"_"+args.label_file))
        
        print("Load file {}".format(os.path.join(args.output_path, "full_ranking"+"_"+args.dataset+"_"+args.result_file)))
        with codecs.open(os.path.join(args.output_path, "full_ranking"+"_"+args.dataset+"_"+args.result_file), "r", "utf-8") as file:
            for line in file.readlines():
                content = line.split('\t')

                q_id.append(content[0]) 
                features.append([float(content[3]),float(content[4])])
                labels.append(1 if content[1] in q2labels[content[0]] else 0)
                
        train(args, q_id, features, labels)
        
    elif args.mode=="infer":
        
        print("Load file {}".format(os.path.join(args.output_path, "full_ranking"+"_"+args.dataset+"_"+args.result_file))) 
        q_id = []
        p_id = []
        features = []
        
        with codecs.open(os.path.join(args.output_path, "full_ranking"+"_"+args.dataset+"_"+args.result_file), "r", "utf-8") as file:
            for line in file.readlines():
                content = line.split('\t')

                q_id.append(content[0]) 
                p_id.append(content[1])
                features.append([float(content[3]),float(content[4])])
        
        scores = inference(args, q_id, p_id, features) 
        
        for q_id, p2score in scores.items():
            sorted_p2score=sorted(p2score.items(), key=lambda x:x[1], reverse = True)
            scores[q_id]=sorted_p2score
        
        with codecs.open(os.path.join(args.output_path, "re_ranking"+"_"+args.dataset+"_"+args.result_file), "w", "utf-8") as file:
            for q_id, p2score in scores.items():
                row_num=0
                for (p_id, score) in p2score:
                    row_num+=1
     
                    ranking = row_num            
                    
                    file.write('\t'.join([q_id, p_id, str(ranking), str(score), "re_ranking"+"_"+args.dataset])+os.linesep)
                    
        print("Produce file {}".format(os.path.join(args.output_path, "re_ranking"+"_"+args.dataset+"_"+args.result_file))) 
                    
    else:
        raise Exception
        
    None


def evaluation(args):
    
    #print("Load file {}".format(os.path.join(args.data_path, args.dataset+"_"+args.label_file)))
    q2labels = label_loader(os.path.join(args.data_path, args.dataset+"_"+args.label_file))
    
    print("Load file {}".format(os.path.join(args.output_path, args.ranking_type+"_"+args.dataset+"_"+args.result_file)))
    
    scores = collections.defaultdict(list)
    with codecs.open(os.path.join(args.output_path, args.ranking_type+"_"+args.dataset+"_"+args.result_file), "r", "utf-8") as file:
        for line in file.readlines():
            content = line.split('\t')
            #scores[content[0]][int(content[2])]=content[1] 
            scores[content[0]].append(content[1]) 
    
    
    print('MRR@{}: {:.4f}'.format(args.topk, mrr(scores, q2labels, args.topk)))
    print('NDCG@{}: {:.4f}'.format(args.topk, ndcg(scores, q2labels, args.topk)))
  
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # overall 
    parser.add_argument("--name", type=str, default='pipeline')
    parser.add_argument("--mode", type=str, default='evaluate') # [train, infer, evaluate]
    parser.add_argument("--ranking_type", type=str, default='full_ranking') # [full_ranking, re_ranking, evaluation]
    parser.add_argument("--dataset", type=str, default='test') # [training, valiation, test]
    parser.add_argument("--topk", type=int, default=100)
    
    # file paths
    parser.add_argument("--data_path", type=str, default='data') 
    parser.add_argument("--output_path", type=str, default='output') 
    parser.add_argument("--passage_file", type=str, default='passages.json')
    parser.add_argument("--query_file", type=str, default='queries.json')
    parser.add_argument("--label_file", type=str, default='labels.json')
    parser.add_argument("--result_file", type=str, default='result.text')
    
    # hyperparameters for RankNet
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--input_size", type=int, default=2)
    parser.add_argument("--hidden_size1", type=int, default=128)
    parser.add_argument("--hidden_size2", type=int, default=128)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    
    # replicability
    parser.add_argument("--replicability", action='store_true')
    parser.add_argument("--random_seed", type=int, default=0)


    args = parser.parse_args()
    
    if args.replicability:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    
    if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
    
    
    print("Start {} {} on the {} set".format(args.mode, args.ranking_type, args.dataset))
    
    tic = datetime.datetime.now()
    
    if args.mode in ["train", "infer"]:
        if args.ranking_type == "full_ranking":
            full_ranking(args)
        elif args.ranking_type=="re_ranking":
            re_ranking(args)
        else:
            raise Exception
            
    elif args.mode == "evaluate":
        evaluation(args)
        
    else:
        raise Exception
        
    toc = datetime.datetime.now()
    
    print("finished within {}".format(toc - tic))
        
    
