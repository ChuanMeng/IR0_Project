import math
import json
import collections


class tfidf(object):
    def __init__(self, passages):
        self.num_p = len(passages)
        self.passages = passages
        self.f = {}     
        self.df = {}    
        self.idf = {}  
        self.init()

    def init(self):
        for p_id, words in self.passages.items():
            tmp = {}
            for word in words:
                tmp[word] = tmp.get(word, 0) + 1  
                
            self.f[p_id]=tmp
            
            for word in tmp.keys():
                self.df[word] = self.df.get(word, 0) + 1
                
        for word, num_p_word in self.df.items():
            self.idf[word] = math.log(self.num_p)-math.log(num_p_word+1)
            

    def sim(self, queries):
        
        scores = collections.defaultdict(dict)
        
        for q_id, q_words in queries.items():
            for p_id, _ in self.passages.items():
                score=0
                for q_word in q_words:
                    if q_word not in self.f[p_id]:
                        continue
                    score += self.f[p_id][q_word]*self.idf[q_word]
                    
                scores[q_id][p_id]=score
        
        return scores  
