import torch
from torch import nn
#from torch.utils.data import Dataset, DataLoader
#from torch.optim import Adam

import pandas as pd
import numpy as np
import os

from utils import MODEL_CLASSES, MODEL_PATH_MAP #SPECIAL_TOKENS

data_path=os.getcwd()+'/../../dataset/'
max_tokenizer_length = 100 #100 #20 
#WiC 데이터는 (SENTENCE1,SENTENCE2) 문장길이가 길어서 maxlength가 넘으면 잘라줬음. 
# 나중에 넘는건 제외하는 식으로 바꿔야함.

model_name = 'koelectra' #'kobert', 'roberta-base', 'koelectra', 'koelectra_QA', 'koelectra_tunib'
config_class, model_class, _ = MODEL_CLASSES[model_name] #config_class, model_class, model_tokenizer
model_path = MODEL_PATH_MAP[model_name]
modelClass_config = config_class.from_pretrained(model_path)

## COLA model ##
class model_COLA(nn.Module):
    def __init__(self):
        super(model_COLA, self).__init__()
        self.model_PLM = model_class.from_pretrained(model_path) #[10,100]*3 -> [10,100,768]
        #self.relu = nn.ReLU() # BERT에서는 GELU 사용.
        self.linear = nn.Linear(768,2) #CLS: 200->2(binary)

    def forward(self, input_ids, token_type_ids, attention_mask): #MyModel에 input(input_ids,token_type_ids,attention_mask) 텐서들이 들어왔을때 forward를 실행.
        iIds = input_ids.long()#.unsqueeze(0).long()
        tok_typeIds = token_type_ids.long()#.unsqueeze(0).long()
        attMask = attention_mask.long()#.unsqueeze(0)
        
        output=self.model_PLM(input_ids=iIds, token_type_ids=tok_typeIds, attention_mask=attMask) #torch.Size([batch_size, max_len, 768])
        output=output['last_hidden_state'][:, 0, :] #CLS token: max_len의 길이 토큰 중 첫번째(0번째) 토큰의 마지막 레이어만 임베딩으로 사용

        #output = self.relu(output)
        output=self.linear(output) #768->2
        
        return output

## Wic model ##
class model_WiC_biSent(nn.Module):
    def __init__(self):
        super(model_WiC_biSent, self).__init__()
        self.model_PLM = model_class.from_pretrained(model_path)
        
        self.linear = nn.Linear(768,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, attention_mask): #MyModel에 input(input_ids,token_type_ids,attention_mask) 텐서들이 들어왔을때 forward를 실행.
        iIds1 = input_ids.long()
        tok_typeIds1 = token_type_ids.long()
        attMask1 = attention_mask.long()

        output=self.model_PLM(input_ids=iIds1, token_type_ids=tok_typeIds1, attention_mask=attMask1) 
        
        output = output['last_hidden_state'][:, 0, :]

        self.sigmoid(output) #output의 tensor 소숫값을 확률값(0~1)로 바꿔줌. (bs,768) -> (bs,768)
        output = self.linear(output) #tensor: 768 -> 1 #linear classifier

        #output = self.softmax(output)
        #print(output.shape)
        
        return output
    
## COPA model ##
class model_COPA_biSent(nn.Module):
    def __init__(self):
        super(model_COPA_biSent, self).__init__()
        self.model_PLM = model_class.from_pretrained(model_path)

        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(768,1) #CLS: 768->1 (model-output concat해서 실질적으론 2개)

    def forward(self, input_ids, token_type_ids, attention_mask): #MyModel에 input(input_ids,token_type_ids,attention_mask) 텐서들이 들어왔을때 forward를 실행.
        
        iIds = input_ids.long() #.unsqueeze(0).long()
        tok_typeIds = token_type_ids.long() #.unsqueeze(0).long()
        attMask = attention_mask.long() #.unsqueeze(0)
        
        output=self.model_PLM(input_ids=iIds, token_type_ids=tok_typeIds, attention_mask=attMask) #token_type_ids=tok_typeIds, 
        #print('output(bert-lhs)[CLS]: ',output['last_hidden_state'][:, 0, :].shape) #CLS token: torch.Size([batch_size, 768])

        #CLS토큰임베딩&LinearFC 사용하는 경우
        output = output['last_hidden_state'][:, 0, :] #cls_embedding: (bs,768)
        self.sigmoid(output) #output의 tensor 소숫값을 확률값(0~1)로 바꿔줌. (bs,768) -> (bs,768)
        output = self.linear(output) #tensor: 768 -> 1 #linear classifier

        return output

## BoolQA model ##
class model_BoolQA(nn.Module):
    def __init__(self):
        super(model_BoolQA, self).__init__()
        self.model_PLM = model_class.from_pretrained(model_path, config=modelClass_config) #[10,100]*3 -> [10,100,768]
        #self.relu = nn.ReLU() # Activation Func.
        self.linear = nn.Linear(768,2) #CLS: 200->2(binary)

    def forward(self, input_ids, token_type_ids, attention_mask): #MyModel에 input(input_ids,token_type_ids,attention_mask) 텐서들이 들어왔을때 forward를 실행.
        iIds = input_ids.long()#.unsqueeze(0).long()
        tok_typeIds = token_type_ids.long()#.unsqueeze(0).long()
        attMask = attention_mask.long()#.unsqueeze(0)

        output=self.model_PLM(input_ids=iIds, token_type_ids=tok_typeIds, attention_mask=attMask) #token_type_ids=tok_typeIds,

        output=output['last_hidden_state'][:, 0, :]

        #self.relu(output)
        output=self.linear(output) #768->2
        
        return output
    
##############################END################

## TODO: WiC model (abandoned) ##
class model_WiC_biSent_abandoned(nn.Module):
    def __init__(self):
        super(model_WiC_biSent_abandoned, self).__init__()
        #self.bert1 = BertModel.from_pretrained('monologg/kobert') #PreTrainedModel(input_ids, token_type_ids, attention_mask): [10,100]*3 -> [10,100,768]
        #self.bert = BertModel.from_pretrained('monologg/kobert')
        self.model_PLM = model_class.from_pretrained(model_path)
        
        #분류에 token embedding을 활용하는 경우
        #self.pooling = nn.AvgPool1d(3)
        self.relu = nn.ReLU() #활성화 함수(0~1): nn.sigmoid, nn.relu(-를 0으로)
        self.sigmoid = nn.Sigmoid()
        self.cosSim = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        #tokenEmbedding2개 붙인거를 linear 하게.       
        self.liearConcat = nn.Linear(768*3,768) #tokenOutput concat할때,.
        self.linear = nn.Linear(768,1) #CLS: 768->2(binary)
        #self.pooling = nn.Linear(768,1)
        self.softmax = nn.Softmax(dim=1) #dim=0(세로합=1),dim=1(가로합=1) #nn.LogSoftmax()

    def forward(self, input_ids, token_type_ids, attention_mask, tokIdx_start, tokIdx_end): #MyModel에 input(input_ids,token_type_ids,attention_mask) 텐서들이 들어왔을때 forward를 실행.
        iIds = input_ids.long()#.unsqueeze(0).long()
        tok_typeIds = token_type_ids.long()#.unsqueeze(0).long()
        attMask = attention_mask.long()#.unsqueeze(0)
        
        output=self.model_PLM(input_ids=iIds, token_type_ids=tok_typeIds, attention_mask=attMask) 
        #print('output(bert-lhs)[CLS]: ',output['last_hidden_state'][:, 0, :].shape) #CLS token: torch.Size([batch_size, 768])
        #print('output shape: ',output['last_hidden_state'].shape, tokIdx_start.shape, tokIdx_end.shape) 
        # #torch.Size([bs, 100, 768]), torch.Size([100])*2개
        
        '''
        #tokenIdx 뽑아서 embeddings에 저장.(mean-pooling)
        embeddings = []
        for idx, (iTok_s,iTok_e) in enumerate(zip(tokIdx_start,tokIdx_end)):
            embedding = []
            #print('token start/end: ',int(iTok_s),int(iTok_e)) #4 4로 똑같다고 할때, 아래와 같음.
            for iTok in range(iTok_s,iTok_e+1):
                embedding.append(output['last_hidden_state'][idx, iTok, :] )
            #print(len(embedding), embedding[0].shape, type(embedding[0])) #list, torch.Size([1, 768])
            embedding = torch.stack([embedding[i] for i in range(len(embedding))])
            #print(embedding.shape, type(embedding)) #torch.Size([1, 768]) <class 'torch.Tensor'>
            embedding = torch.mean(embedding, dim=0) #mean-pooling 
            #print(embedding.shape, type(embedding)) #torch.Size([768]) <class 'torch.Tensor'>
            embeddings.append(embedding)
        #print('embeddings: ',type(embeddings), len(embeddings), embeddings[0].shape) #<class 'list'> 50 torch.Size([768])
        output = torch.stack(embeddings, dim=0) #list[torch[768]]*100 -> 
        #print('embeddings_stacked: ',output.shape) #torch.Size([50, 768])
        '''
        #tokenIdx 뽑아서 embeddings에 저장.(concat, +-1)
        embeddings = []
        for idx, (iTok_s,iTok_e) in enumerate(zip(tokIdx_start,tokIdx_end)):
            embedding = []
            #print('token start/end: ',int(iTok_s),int(iTok_e)) #4 4로 똑같다고 할때, 아래와 같음.
            for iTok in range(iTok_s-1,iTok_s+2):
                embedding.append(output['last_hidden_state'][idx, iTok, :] )
            #print(len(embedding), embedding[0].shape, type(embedding[0])) #list, torch.Size([1, 768])
            
            embedding = torch.cat(embedding, dim=0) #Token전략: +-1씩 concat후, linearPooling
            #print('cat:',embedding.shape, type(embedding))
            embedding = self.liearConcat(embedding) # 3*torch.Size([768]) -> torch.Size([2304])
            #print('lincat:',embedding.shape, type(embedding))
            embedding = torch.stack([embedding[i] for i in range(len(embedding))])
            #print('stack:',embedding.shape, type(embedding)) #torch.Size([1, 768]) <class 'torch.Tensor'>
            
            #embedding = torch.mean(embedding, dim=0) #Token전략: mean-pooling 
            #print('pooling:',embedding.shape, type(embedding)) #torch.Size([768]) <class 'torch.Tensor'>
            embeddings.append(embedding)
        #print('embeddings: ',type(embeddings), len(embeddings), embeddings[0].shape) #<class 'list'> 50 torch.Size([768])
        output = torch.stack(embeddings, dim=0) #list[torch[768]]*50 -> [50, 768]
        #print('embeddings_stacked: ',output.shape) #torch.Size([50, 768])


        
        #print('concatLinear: ',output.shape, output[0].shape, output[0][:20]) #
        #self.sigmoid(output)
        self.relu(output)
        #print('activation Func: ',output.shape, output[0].shape, output[0][:20]) #torch.Size([50,768]) tensor(0.0005...

        #output = self.linear(output)
        #print('embeddings_reluLinear: ',output.shape) #torch.Size([50, 1])

        
        #output = self.softmax(output)
        #print(output.shape)
        
        return output
    
    