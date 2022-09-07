import torch.utils.data as data
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import codecs
import json
import collections



def get_format_data(args):
    
    q_id = []
    x = []
    y = []
    
    
    label = json.load(open("data/"+args.dataset+"_labels.json", 'r'))
    
    with codecs.open(args.dataset+"_full_ranking.text", "r", "utf-8") as file:
        for line in file.readlines():
            content = line.split('\t')

            q_id.append(content[0]) 
            x.append([float(content[3]),float(content[4])])
            y.append(1 if content[1] in label[content[0]] else 0)
            
            
            #print(content[0],[float(content[3]),float(content[4])],1 if content[1] in label[content[0]] else 0)
            
    return q_id, x, y
    
  
def get_pair_passage_data(q_id, x, y):
    pairs = []
    tmp_pair1 = []
    tmp_pair2 = []
    
    for i in range(0, len(q_id) - 1):
        for j in range(i + 1, len(q_id)):         
            if q_id[i] != q_id[j]:
                break

            if (q_id[i] == q_id[j]) and (y[i] != y[j]):

                if y[i] > y[j]:
                    pairs.append([i,j])
                    tmp_pair1.append(x[i])
                    tmp_pair2.append(x[j])
                    
                else:
                    pairs.append([j,i])
                    tmp_pair1.append(x[j])
                    tmp_pair2.append(x[i])
    

    tensor_pair1 = torch.tensor(tmp_pair1)
    tensor_pair2 = torch.tensor(tmp_pair2)
    
    print('found {} passage pairs'.format(len(pairs)))
    return len(pairs), tensor_pair1, tensor_pair2


class Dataset(data.Dataset):

    def __init__(self, args):
        q_id, x, y = get_format_data(args)
        
        self.pair_num, self.tensor_pair1, self.tensor_pair2 = get_pair_passage_data(q_id, x, y)

    def __getitem__(self, index):
        return self.tensor_pair1[index], self.tensor_pair2[index]

    def __len__(self):
        return self.pair_num


def get_loader(args, shuffle, num_workers):
    dataset = Dataset(args)
    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size = args.batch_size,
        shuffle = shuffle,
        num_workers=num_workers
    )
    return data_loader



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RankNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(
            # layer-1
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            # layer-2
            nn.Linear(hidden_size1,hidden_size2),
            nn.ReLU(),
            # layer-out
            nn.Linear(hidden_size2, output_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        result_1 = self.model(input_1) 
        result_2 = self.model(input_2) 
        pred = self.sigmoid(result_1 - result_2) 
        return pred

    def predict(self, input):
        result = self.model(input)
        return result



def train(args):
    


    model = RankNet(args.input_size, args.hidden_size1, args.hidden_size2, args.output_size).to(device)
 
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    data_loader = get_loader(args, True, 1)
    total_step = len(data_loader)

    for epoch in range(args.epochs):
        for i, (data1, data2) in enumerate(data_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            label_size = data1.size()[0]
            pred = model(data1, data2)
            loss = criterion(pred, torch.from_numpy(np.ones(shape=(label_size, 1))).float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step, loss.item()))

    torch.save(model.state_dict(), 'ranknet.ckpt')


def inference(args):

    
    model = RankNet(args.input_size, args.hidden_size1, args.hidden_size2, args.output_size).to(device)
    model.load_state_dict(torch.load('ranknet.ckpt'))
    model.eval()
    
    q_id = []
    p_id = []
    x = []
    
    with torch.no_grad():
        with codecs.open(args.dataset+"_full_ranking.text", "r", "utf-8") as file:
            for line in file.readlines():
                content = line.split('\t')

                q_id.append(content[0]) 
                p_id.append(content[1])
                x.append([float(content[3]),float(content[4])])
                print(content[0],content[1],[float(content[3]),float(content[4])])

        tensor_x = torch.tensor(x)
        y = model.predict(tensor_x)  # [query_num * topk]
    
    
    
    logger= collections.defaultdict(dict)
    
    for combination in zip(q_id,p_id,y):
        logger[combination[0]][combination[1]]=combination[2].item()
            
    #print(logger)
    
    for q_id, p_id_score in logger.items():
        sorted_p_id_score=sorted(p_id_score.items(), key=lambda x:x[1], reverse = True)
        
        logger[q_id]=sorted_p_id_score
        
        #print("=====")
        #print(q_id)
        #print()
        #print(p_id_score)
        #print()
        #print(sorted_p_id_score)
        
    
    with codecs.open(args.dataset+"_re_ranking.text", "w", "utf-8") as file:
        for q_id, p_id_score in logger.items():
            row_num=0
            for p_id, score in p_id_score:
                row_num+=1
 
                ranking =  row_num            
                
                file.write('\t'.join([q_id, p_id, str(ranking), str(score), args.ranking+"_"+args.dataset])+os.linesep)
        
    

    """
    top_p_indexs = np.argsort(-predicted_y, 1)[:, :args.topk]
    
    
        
    """  
