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

    passages_tokenised={}
    p_id2len={}
    
    for p_id in passages.keys():
        passages_tokenised[p_id]=process_text(passages[p_id])
        p_id2len[p_id]=len(passages_tokenised[p_id])
        
        assert len(passages_tokenised[p_id])==len(process_text(passages[p_id]))

    print("num_passage:", len(passages))

    assert len(p_id2len)==len(passages)
    return passages_tokenised, p_id2len


def process_query(queries):
    
    
    queries_tokenised={}
    
    for q_id in queries.keys():
        queries_tokenised[q_id]= process_text(queries[q_id])
    
    print("num_query:", len(queries_tokenised))
    
    return queries_tokenised  
