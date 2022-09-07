#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:50:48 2022

@author: c.meng
"""

import numpy as np


def ndcg(outputs, labels, k):
    """
    Compute normalized discounted cumulative gain.
    :return: mean average precision [a float value]
    """
    assert len(outputs) == len(labels)
    #assert len(output)s >= k, 'NDCG@K cannot be computed, invalid value of K.'


    NDCG = 0
    
    for q_id, p_ids in outputs.items():
        DCG_pred = 0
        num_gt_passage = len(labels[q_id])
        
        for position, p_id in p_ids.items():
            
            if position > k:
                break
            if p_id in labels[q_id]:
                DCG_pred += 1 / np.log2(position + 1)
                
        
        DCG_gt = 0
        for j in range(num_gt_passage):
            if j+1 > k:
                break
            DCG_gt += 1 / np.log2(j+ 1 + 2)
        NDCG += DCG_pred / DCG_gt
                

    return NDCG / len(outputs)


def mrr(outputs, labels, k):
    """
    Compute mean reciprocal rank.
    :return: mean reciprocal rank [a float value]
    """
    assert len(outputs) == len(labels)
    
    reciprocal_rank = 0
    
    for q_id, p_ids in outputs.items():
        for position, p_id in p_ids.items():
            if position > k:
                break
            
            if p_id in labels[q_id]:
                reciprocal_rank += 1.0 / (position)
                break

    return reciprocal_rank / len(outputs)
