
"""
Created on Tue Aug 30 17:50:48 2022

@author: c.meng
"""

import numpy as np


def mrr(scores, labels, k):
    # only consider the position of the first passage that hit the groud truth 
    assert len(scores) == len(labels)
    
    reciprocal_rank = 0
    
    for q_id, p_ids in scores.items():
        
        for index, p_id in enumerate(p_ids):
            if index == k:
                break
            
            if p_id in labels[q_id]:
                reciprocal_rank += 1.0 / (index+1)
                break 

    return reciprocal_rank / len(scores)


def ndcg(scores, labels, k):
    assert len(scores) == len(labels)
    NDCG = 0
    
    for q_id, p_ids in scores.items():
        DCG_pred = 0
        num_gt_passage = len(labels[q_id])
        assert len(p_ids)>= k # 'NDCG@K cannot be computed, invalid value of K.'
        
        for index, p_id in enumerate(p_ids):   
            if index == k:
                break
            if p_id in labels[q_id]:
                DCG_pred += 1 / np.log2(index + 1 + 1)
                
        
        DCG_gt = 0
        for index in range(num_gt_passage):
            if index == k:
                break
            DCG_gt += 1 / np.log2(index+ 1 + 1)
        NDCG += DCG_pred / DCG_gt
                

    return NDCG / len(scores)