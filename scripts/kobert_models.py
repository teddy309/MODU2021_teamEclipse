import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import pandas as pd
import numpy as np
import os

from PIL import Image
import copy

from utils import MODEL_CLASSES, MODEL_PATH_MAP #SPECIAL_TOKENS

data_path=os.getcwd()+'/../../dataset/'
max_tokenizer_length = 100 #100 #20 
#WiC 데이터는 (SENTENCE1,SENTENCE2) 문장길이가 길어서 maxlength가 넘으면 잘라줬음. 
# 나중에 넘는건 제외하는 식으로 바꿔야함.

#config_class, model_class, _ = MODEL_CLASSES['kobert'] #
model_name = 'koelectra' #'kobert', 'roberta-base', 'koelectra', 'koelectra_QA', 'koelectra_tunib'
config_class, model_class, _ = MODEL_CLASSES[model_name] #config_class, model_class, model_tokenizer
model_path = MODEL_PATH_MAP[model_name]
modelClass_config = config_class.from_pretrained(model_path)

#monologg/KoBERT-nsmc
class model_COLA(nn.Module):
    def __init__(self):
        super(model_COLA, self).__init__()
        self.model_PLM = model_class.from_pretrained(model_path) #[10,100]*3 -> [10,100,768]
        self.relu = nn.ReLU() # 
        self.linear = nn.Linear(768,2) #CLS: 200->2(binary)

    def forward(self, input_ids, token_type_ids, attention_mask): #MyModel에 input(input_ids,token_type_ids,attention_mask) 텐서들이 들어왔을때 forward를 실행.
        iIds = input_ids.long()#.unsqueeze(0).long()
        tok_typeIds = token_type_ids.long()#.unsqueeze(0).long()
        attMask = attention_mask.long()#.unsqueeze(0)
        
        output=self.model_PLM(input_ids=iIds, token_type_ids=tok_typeIds, attention_mask=attMask) #torch.Size([batch_size, max_len, 768])
        #print('output(bert-lhs)[CLS]: ',output['last_hidden_state'][:, 0, :].shape) #CLS token: torch.Size([batch_size, 768])
        output=output['last_hidden_state'][:, 0, :] #CLS token: max_len의 길이 토큰 중 첫번째(0번째) 토큰의 마지막 레이어만 임베딩으로 사용
        #torch.Size([batch_size, 768])

        self.relu(output)
        output=self.linear(output) #768->1
        
        return output


#monologg/KoBERT-nsmc
class model_WiC_uniSent(nn.Module):
    def __init__(self):
        super(model_WiC_uniSent, self).__init__()
        #self.bert = BertModel.from_pretrained('monologg/kobert') #PreTrainedModel(input_ids, token_type_ids, attention_mask): [10,100]*3 -> [10,100,768]
        self.model_PLM = model_class.from_pretrained(model_path)

        self.relu = nn.ReLU() # 
        self.softmax = nn.Softmax(dim=1) #dim=0(세로합=1),dim=1(가로합=1) #nn.LogSoftmax()

        #Token Embedding 사용하는 경우.
        #self.pooling = nn.AvgPool1d(3) #token위치에서 pooling 방법.
        #self.cosSim = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        #CLS토큰&LinearFC 사용하는 경우
         #self.linear = nn.Linear(768,2) #CLS: 768->2(binary)
        self.fc1 = nn.Linear(768,768*4)
        self.fc2 = nn.Linear(768*4,768)
        self.fc3 = nn.Linear(768, 192)
        self.fc4 = nn.Linear(192, 2)

    def forward(self, input_ids, token_type_ids, attention_mask, tok1_StartIdx, tok2_StartIdx): #MyModel에 input(input_ids,token_type_ids,attention_mask) 텐서들이 들어왔을때 forward를 실행.
        iIds = input_ids.long()#.unsqueeze(0).long()
        tok_typeIds = token_type_ids.long()#.unsqueeze(0).long()
        attMask = attention_mask.long()#.unsqueeze(0)
        
        #print('bert-layer input(iIds,tok_typeIds,attMask): ',iIds.shape, tok_typeIds.shape, attMask.shape) #
        #print('iIds: ',iIds) #torch.Size([10, 100]) when bs=10
        
        output=self.model_PLM(input_ids=iIds, token_type_ids=tok_typeIds, attention_mask=attMask) #token_type_ids=tok_typeIds, 
        #print('output(bert-lhs)[CLS]: ',output['last_hidden_state'][:, 0, :].shape) #CLS token: torch.Size([batch_size, 768])

        '''
        #token embedding 2개에서 cos-similarity로 분류하는 경우.
        embed_list = [] #len=2, [torch.Size([bs, bs, 768]) * 2개]
        for idx in [tok1_StartIdx, tok2_StartIdx]:
            embedding = output['last_hidden_state'][:, idx, :] 
            #s1_start, s2_start embedding, 여유되면 idx+-1로 mean-pooling

            #self.pooling(embedding)
            self.relu(embedding)
            #embedding=self.linear(embedding)
            embed_list.append(embedding)

        #print('pooled-emb shape: ',embedding.shape) #torch.Size([bs, bs, 768])
        
        output = self.cosSim(embed_list[0], embed_list[1]) #cosSim(torch.Size([bs, bs, 768])*2개) -> torch.Size([bs, 768])
        #print(output.shape) #torch.Size([100, 768])
        output=self.linear(output) #torch.Size([bs, 2])
        #print(output.shape) #torch.Size([100, 2])
        '''

        #CLS토큰임베딩&LinearFC 사용하는 경우
        output = output['last_hidden_state'][:, 0, :] #cls_embedding
        output = self.fc1(output) #tensor: 768 -> 768*4
        output = self.fc2(output) #tensor: 768*4 -> 768
        output = self.fc3(output) #tensor: 768 -> 192
        output = self.fc4(output) #tensor: 192 -> 2

        output = self.softmax(output)

        return output

#monologg/KoBERT-nsmc
class model_WiC_biSent(nn.Module):
    def __init__(self):
        super(model_WiC_biSent, self).__init__()
        #self.bert1 = BertModel.from_pretrained('monologg/kobert') #PreTrainedModel(input_ids, token_type_ids, attention_mask): [10,100]*3 -> [10,100,768]
        #self.bert = BertModel.from_pretrained('monologg/kobert')
        self.model_PLM = model_class.from_pretrained(model_path)
        
        #분류에 token embedding을 활용하는 경우
        #self.pooling = nn.AvgPool1d(3)
        #self.relu = nn.ReLU() #활성화 함수(0~1): nn.sigmoid, nn.relu(-를 0으로)
        #self.sigmoid = nn.Sigmoid()
        self.cosSim = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        #tokenEmbedding2개 붙인거를 linear 하게.       
        self.linear = nn.Linear(768*2,2) #CLS: 768->2(binary)
        #self.pooling = nn.Linear(768,1)
        self.softmax = nn.Softmax(dim=1) #dim=0(세로합=1),dim=1(가로합=1) #nn.LogSoftmax()
        
        #CLS토큰을 사용하는 경우(concat/mean-pooling/단일토큰 등)
        #flatEmb = 768*2*max_tokenizer_length
        #self.fc0 = nn.Linear(768*2,flatEmb)
        #self.fc1 = nn.Linear(768*2,768*4) #concat했을때
        #self.fc2 = nn.Linear(768*4,768)
        #self.fc3 = nn.Linear(768, 192)
        #self.fc4 = nn.Linear(192, 2)

    def forward(self, input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2, tok1_StartIdx, tok2_StartIdx): #MyModel에 input(input_ids,token_type_ids,attention_mask) 텐서들이 들어왔을때 forward를 실행.
        iIds1 = input_ids1.long()#.unsqueeze(0).long()
        tok_typeIds1 = token_type_ids1.long()#.unsqueeze(0).long()
        attMask1 = attention_mask1.long()#.unsqueeze(0)

        iIds2 = input_ids2.long()#.unsqueeze(0).long()
        tok_typeIds2 = token_type_ids2.long()#.unsqueeze(0).long()
        attMask2 = attention_mask2.long()#.unsqueeze(0)
        
        output1=self.model_PLM(input_ids=iIds1, token_type_ids=tok_typeIds1, attention_mask=attMask1) 
        output2=self.model_PLM(input_ids=iIds2, token_type_ids=tok_typeIds2, attention_mask=attMask2)
        #print('output(bert-lhs)[CLS]: ',output['last_hidden_state'][:, 0, :].shape) #CLS token: torch.Size([batch_size, 768])
        
        '''
        #여유되면 tokenIdx를 따와서 넣어주기.(mean-pooling, concat, )
        embed_list = []
        for idx in tokStartIdx_list:
            embedding1 = output1['last_hidden_state'][:, idx, :] 
            #s1_start, s2_start embedding, 여유되면 idx+-1로 mean-pooling

            self.relu(embedding)
            #embedding=self.linear(embedding)
            embed_list.append(embedding)
        '''
        #s1_start, s2_start embedding, 여유되면 idx+-1로 mean-pooling
        idx1, idx2 = tok1_StartIdx.unsqueeze(1).long(), tok2_StartIdx.unsqueeze(1).long() 
        #idx1, idx2 = tok1_StartIdx.long(), tok2_StartIdx.long() 
        #print('idx1:', idx1.shape, 'idx2:', idx2.shape, type(idx1), type(idx2)) #torch.Size([50, 1]) torch.Size([50, 1]) <class 'torch.Tensor'> <class 'torch.Tensor'>
        #print(idx1.shape, idx2.shape) #int
        
        output1 = output1['last_hidden_state']
        output2 = output2['last_hidden_state']
        #print('output(1,2)[LHS] shape:',output1.shape, output2.shape) #torch.Size([bs, maxLen, 768]): torch.Size([50, 100, 768])
        
        
        embedding1, embedding2 = [], []
        for batchIdx, (tok_i1, tok_i2) in enumerate(zip(idx1,idx2)):
            embedding1.append(output1[batchIdx, tok_i1, :].squeeze())
            embedding2.append(output2[batchIdx, tok_i2, :].squeeze())
            #output1 = output1[:, tok_i1, :]
            #output2 = output2[:, tok_i2, :]
        embedding1 = torch.stack([embedding1[i] for i in range(len(embedding1))]) #torch.tensor(embedding1) #
        embedding2 = torch.stack([embedding2[i] for i in range(len(embedding2))]) #torch.tensor(embedding2) #
        #print(type(embedding1), type(embedding2))
        #print('embedding(1,2) len: ',len(embedding1), len(embedding2), type(embedding1), embedding1[0].shape) #1 1 <class 'list'> torch.Size([50, 768]) -> 50 50 <class 'list'> torch.Size([1, 768])
        #print('embedding(2) shape: ', embedding2.shape, embedding2[0].shape) # torch.Size([50, 768]) torch.Size([768])

        #self.pooling(embedding1)
        #self.pooling(embedding2)
        #self.sigmoid(embedding1) #self.relu(embedding1)
        #self.sigmoid(embedding2) #self.relu(embedding2)

        
        #두 토큰임베딩을 concat한 경우.
        embedding = torch.cat([embedding1,embedding2],dim=1) # torch.Size([bs, 768])*2개 -> 768*2 #torch.Size([bs, maxLen*2, 768])
        #print('(cat)embedding shape: ',embedding1.shape,'+',embedding2.shape,' = ',embedding.shape) #torch.Size([bs, maxLen*2, 768]) torch.Size([bs, maxLen, 768]) torch.Size([bs, maxLen, 768])
        output=self.linear(embedding) #embedding=self.linear(embedding)
        #output = self.fc1(embedding) #tensor: 768*2 -> 768*4
        #output = self.fc2(output) #tensor: 768*4 -> 768
        #output = self.fc3(output) #tensor: 768 -> 192
        #output = self.fc4(output) #tensor: 192 -> 2
        output = self.softmax(output)
        

        '''
        #embedding1,2가 0~1 범위인지 검증. 아니면 softmax
        #for i in range(len(embedding1)):
        #    assert embedding1[i] == abs(embedding1[i]), f'{i} is not plus value'
        #for i in range(len(embedding2)):
        #    assert embedding2[i] == abs(embedding2[i])
        similarity = self.cosSim(embedding1, embedding2) #[bs,768]*2개 -> [bs,]
        print('similarity: ', similarity.shape, similarity, similarity.dtype)
        output = similarity.unsqueeze(1)
        #output = self.softmax(similarity.unsqueeze(1))
        '''


        #u = self.pooling(embedding1) #tensor: 768 -> 1
        #v = self.pooling(embedding2) #tensor: 768 -> 1
        #loss = abs(u-v)
        #output = torch.cat([u,v,loss],dim=1)
        #print(output, self.softmax(output))
        #print(output.shape, self.softmax(output).shape) #torch.Size([50, 1]) torch.Size([50, 1])
        
        #output = self.softmax(output)
        print(output.shape)
        
        return output
    
#monologg/KoBERT-nsmc
class model_WiC_biSent1(nn.Module):
    def __init__(self):
        super(model_WiC_biSent1, self).__init__()
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
    
    
#monologg/KoBERT-nsmc
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


#monologg/KoBERT-nsmc
class model_BoolQA(nn.Module):
    def __init__(self):
        super(model_BoolQA, self).__init__()
        self.model_PLM = model_class.from_pretrained(model_path, config=modelClass_config) #[10,100]*3 -> [10,100,768]
        self.relu = nn.ReLU() # Activation Func.
        self.linear = nn.Linear(768,2) #CLS: 200->2(binary)

    def forward(self, input_ids, token_type_ids, attention_mask, sepIdx_s, sepIdx_e): #MyModel에 input(input_ids,token_type_ids,attention_mask) 텐서들이 들어왔을때 forward를 실행.
        iIds = input_ids.long()#.unsqueeze(0).long()
        tok_typeIds = token_type_ids.long()#.unsqueeze(0).long()
        attMask = attention_mask.long()#.unsqueeze(0)

        output=self.model_PLM(input_ids=iIds, token_type_ids=tok_typeIds, attention_mask=attMask) #token_type_ids=tok_typeIds, 

        #last_hidden_state의 span을 pooling해서 리스트(bs*1)로 반환.
        qEmbeddings = [] #span-pooled list
        for batchIdx, (sep_s, sep_e) in enumerate(zip(sepIdx_s,sepIdx_e)):
            qEmbedding=[]
            for idx in range(sep_s+1,sep_e):
                qEmbedding.append(output['last_hidden_state'][batchIdx, idx, :].squeeze()) #CLS token
            qEmbedding = torch.stack([qEmbedding[i] for i in range(len(qEmbedding))]) #torch.tensor(embedding1)
            qEmbedding = torch.mean(qEmbedding, dim=0) #mean-pooling #mean,max,concat 등 시도해보기.
            qEmbeddings.append(qEmbedding)
            #print(batchIdx, qEmbedding.shape)
        #print('qEmbedding:', len(qEmbeddings), type(qEmbeddings), type(qEmbeddings[0])) #20 <class 'list'> <class 'list'>
        output = torch.stack([qEmbeddings[i] for i in range(len(qEmbeddings))]) #torch.tensor(qEmbeddings)
        #print(output.shape) #torch.Size([20, 768])

        self.relu(output)
        output=self.linear(output) #768->1
        
        return output
    
    '''
    def get_sep_embedding(self, input_ids, sequence_output):
        batch_size = input_ids.size(0)
        sep_idx = (input_ids == 3).sum(1) #self.sep_id=3
        sep_embedding = sequence_output[torch.arange(batch_size), sep_idx]
        return sep_embedding
    '''
