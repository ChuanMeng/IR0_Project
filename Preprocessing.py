#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:52:12 2022

@author: c.meng
"""

import math
import numpy as np
import re
import string



def process_text(text):

    return text.split()


def process_passage(passages):

    passages_tokenised=[]
    
    p_id2index = {}
    p_index2id={}
    p_id2len={}
    
    for p_index, p_id in enumerate(passages):
        passages_tokenised.append(process_text(passages[p_id]))
        p_id2index[p_id] = p_index
        p_index2id[p_index] = p_id
        
        p_id2len[p_id]=len(passages_tokenised[p_index])
        assert len(passages_tokenised[p_index])==len(process_text(passages[p_id]))


    print("num_passage:", len(passages))

    assert len(p_id2index)==len(p_index2id)==len(p_id2len)==len(passages)
    return passages_tokenised, p_id2index, p_index2id, p_id2len


def process_query(p_id2index, queries):

    queries_tokenised=[]
    q_index2id={}
    
    for q_index, q_id in enumerate(queries.keys()):
        queries_tokenised.append(process_text(queries[q_id]))
        q_index2id[q_index]=q_id
        
        #label = np.zeros(len(p_id2index))
        
        
        #for p_id in queries["labels"][q_id]:
            #label[p_id2index[p_id]] = 1
            
        #labels.append(label)
    #labels = np.array(labels) # [query_num, passage_num]

    
    print("num_query:", len(queries_tokenised))
    
    return queries_tokenised, q_index2id   
