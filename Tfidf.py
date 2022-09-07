import math
import json



class tfidf(object):
    def __init__(self, docs):
        self.D = len(docs)
        self.docs = docs
        self.f = []     
        self.df = {}    
        self.idf = {}  
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  
            self.f.append(tmp) 
            
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
                
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D)-math.log(v+1)
            
        json_f = {"idf": self.idf, "f": self.f}
        json_f = json.dumps(json_f)
        

    def sim(self, query, index):
        
        score = 0
        
        for word in query:
            if word not in self.f[index]:
                continue
            
            #tf=/len(self.docs[index])
            score += self.f[index][word]*self.idf[word]
            
        return score

    def simall(self, query):
        scores = []
        
        for index in range(self.D):
            score = self.sim(query, index)
            scores.append(score)
            
        return scores  # [passage_nums]
