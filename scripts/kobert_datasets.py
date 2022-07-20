import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import pandas as pd
import numpy as np
import os

from PIL import Image
import copy

#from data.getIndex import getIdxTopN
#from sklearn.metrics import matthews_corrcoef

##
#from transformers import AutoModel, AutoTokenizer, AutoModelWithLMHead #klue/roberta, Beomi/KcBERT
#from transformers import ElectraTokenizer #monologg/koElectra
#from transformers import BertModel #BertTokenizer, BertModel #monologg/kobert
#from tokenization_kobert import KoBertTokenizer #monologg/kobert
#from transformers import T5Model, T5Tokenizer, T5ForConditionalGeneration #kolang-t5-base

#from utils import compute_metrics, MCC, get_label, MODEL_CLASSES, SPECIAL_TOKENS
from utils import MODEL_CLASSES, MODEL_PATH_MAP 
from utils import TOKEN_MAX_LENGTH, getTokLength #SPECIAL_TOKENS

#config_class, model_class, model_tokenizer = MODEL_CLASSES['kobert'] #
model_name = 'koelectra' #'kobert', 'roberta-base', 'koelectra', 'koelectra_QA', 'koelectra_tunib'
_1, _2, model_tokenizer = MODEL_CLASSES[model_name] #config_class, model_class, model_tokenizer
model_path = MODEL_PATH_MAP[model_name]

data_path=os.getcwd()+'/../../dataset/'
max_tokenizer_length = 300#100 #20 #WiC 데이터는 (SENTENCE1,SENTENCE2) 문장길이가 길어서 maxlength가 넘으면 잘라줬음. 나중에 넘는건 제외하는 식으로 바꿔야함.
#max_tokenizer_length = TOKEN_MAX_LENGTH['WiC']

##monologg/KoBERT-nsmc##
class COLA_dataset(Dataset): 
    def __init__(self, data_filename): #str: path of csv
        super(COLA_dataset,self).__init__()
        self.data = pd.read_csv(os.path.join(os.getcwd(),data_filename), sep="\t")#
        self.tokenizer = model_tokenizer.from_pretrained(model_path)
    
    def __len__(self): #return size of dataset(row*column)
        return len(self.data)
        
    def __getitem__(self, item): #fetch data sample(row) for given key(item)
        sentData = self.data.iloc[item,:]
            
        tok = self.tokenizer(sentData['sentence'], padding="max_length", max_length=20, truncation=True) #PreTrainedTokenizer.__call__(): str -> tensor(3*400)
        
        label=sentData['acceptability_label']

        input_ids=torch.LongTensor(tok["input_ids"]) #input token index in vacabs
        token_type_ids=torch.LongTensor(tok["token_type_ids"]) #segment token index 
        attention_mask=torch.LongTensor(tok["attention_mask"]) #boolean: masking attention(0), not masked(1)
            
        return input_ids, token_type_ids, attention_mask, label #tensor(400), tensor(400), tensor(400), int

##monologg/KoBERT-nsmc## 
class WiC_uniSentence(Dataset): 
    def __init__(self, data_filename): #str: path of csv
        super(WiC_uniSentence,self).__init__()
        self.data = pd.read_csv(os.path.join(os.getcwd(),data_filename), sep="\t")#[:100] #data: ~800개까진 OOM 안뜸.
        #self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert') #monologg/kobert
        self.tokenizer = model_tokenizer.from_pretrained(model_path)
    
    def __len__(self): #return size of dataset(row*column)
        return len(self.data)
    
    def __name__(self): #return size of dataset(row*column)
        return 'WiC_uniSentence'
    
    def __getitem__(self, item): #fetch data sample(row) for given key(item)
        sentData = self.data.iloc[item,:]
        #print(len(sentData), type(sentData), sentData) #4, Seires, (csv labels: )

        #print(len(sentData['SENTENCE1']), sentData['SENTENCE1']) #
        #print(len(sentData['SENTENCE2']), sentData['SENTENCE2']) #

        sent1 = sentData['SENTENCE1'][:max_tokenizer_length]
        sent2 = sentData['SENTENCE2'][:max_tokenizer_length]
        
        input_str = sent1+self.tokenizer.sep_token+sent2 #[CLS:2]+S1+[SEP:3]+S2+[SEP]+[PAD:1]
        tok = self.tokenizer(input_str, padding="max_length", max_length=max_tokenizer_length*2, truncation=True)
        #print('tokenized:',self.tokenizer.tokenize(input_str))
        #print('origin sentence:',self.tokenizer.decode(tok['input_ids']))
        #print('token_set:',tok) #(input_ids, token_type_ids, attention_mask)

        label = sentData['ANSWER']
        label_int = 0
        if label==True:
            label_int=1
        else:
            label_int=0
        #print('label:', label, label_int)

        s1_from, s1_to = sentData['start_s1'], sentData['end_s1']
        s2_from, s2_to = sentData['start_s2'], sentData['end_s2']
        s1_offset = len(sentData['SENTENCE1'])+5 #Sent1+[SEP]
        position = [s1_from, s1_to, s1_offset+s2_from, s1_offset+s2_to]
        #print('s1:',position[:2],', s2:',position[2:]) #print('s1:',s1_from, s1_to,', s2:',s2_from, s2_to)
        #print(input_str[position[0]:position[1]], input_str[position[2]:position[3]])

        #get start position's token indexs(for input_str)
        tokStartIdx = []
        for pos in [position[0], position[2]]:
            idx=1 #[CLS]=5
            ctr=5
            for curTok in tok['input_ids']:#while i<pos:
                curTok_org = self.tokenizer.decode(curTok)
                len_tmp=len(curTok_org)
                if curTok==2: #[CLS]
                    #print(pos,' - ',idx,len_tmp,type(curTok_org),curTok_org)
                    continue
                elif ctr>=pos:
                    tokStartIdx.append(idx)
                    break
                else:
                    ctr+=len_tmp
                    idx+=1
                    #print(pos,' - ', idx,ctr,len_tmp, type(curTok_org),curTok_org)
        #print('token StartIdx:',tokStartIdx) #잘 안골라지는데 띄어쓰기 때문인거 같음.
        
        input_ids=torch.LongTensor(tok["input_ids"]) #input token index in vacabs
        token_type_ids=torch.LongTensor(tok["token_type_ids"]) #segment token index 
        attention_mask=torch.LongTensor(tok["attention_mask"]) #boolean: masking attention(0), not masked(1)
            
        return input_ids, token_type_ids, attention_mask, label_int, tokStartIdx[0], tokStartIdx[1] 
        #return type(len): tensor(2*tokMaxLen), tensor(2*tokMaxLen), tensor(2*tokMaxLen), tensor(int), tensor(int), tensor(int)
        #current len: 200, 200, 200, 1, 1, 1

##monologg/KoBERT-nsmc##
class WiC_biSentence(Dataset): 
    def __init__(self, data_filename): #str: path of csv
        super(WiC_biSentence,self).__init__()
        self.data = pd.read_csv(os.path.join(os.getcwd(),data_filename), sep="\t") #TOKEN_MAX_LENGTH:256 [:256] 
        self.tokenizer = model_tokenizer.from_pretrained(model_path)
        self.max_tokenizer_length = TOKEN_MAX_LENGTH['WiC']
    
    def __len__(self): #return size of dataset(row*column)
        return len(self.data)
        
    def __name__(self): #return size of dataset(row*column)
        return 'WiC_biSentence'

    def __getitem__(self, item): #fetch data sample(row) for given key(item)
        sentData = self.data.iloc[item,:]
        #print(len(sentData['SENTENCE1']), sentData['SENTENCE1']) #20 사장은 새로운 기계를 공장에 앉혔다.
        #print(len(sentData['SENTENCE2']), sentData['SENTENCE2']) #22 사원들은 사장의 행보를 예측하지 못했다.
        
        sent1 = sentData['SENTENCE1'][:self.max_tokenizer_length]
        sent2 = sentData['SENTENCE2'][:self.max_tokenizer_length]

        tok1 = self.tokenizer(sent1, padding="max_length", max_length=self.max_tokenizer_length, truncation=True) #PreTrainedTokenizer.__call__(): str -> tensor(3*400)
        tok2 = self.tokenizer(sent2, padding="max_length", max_length=self.max_tokenizer_length, truncation=True)
        #print('tokenized_1:',self.tokenizer.tokenize(sent1)) #['▁사장은', '▁새로운', '▁기계', '를', '▁공장', '에', '▁', '앉', '혔다', '.']
        #print('tokenized_2:',self.tokenizer.tokenize(sent2)) #['▁사', '원', '들은', '▁사장', '의', '▁행보', '를', '▁예측', '하지', '▁못했다', '.']
        
        #print('origin sent1:',self.tokenizer.decode(tok1['input_ids'])) #[CLS] 사장은 새로운 기계를 공장에 앉혔다.[SEP]
        #print('origin sent2:',self.tokenizer.decode(tok2['input_ids'])) #[CLS] 사원들은 사장의 행보를 예측하지 못했다.[SEP]
        
        label = sentData['ANSWER'] #True/False
        s1_from, s1_to = sentData['start_s1'], sentData['end_s1']
        s2_from, s2_to = sentData['start_s2'], sentData['end_s2']

        label_int = 0
        if label==True:
            label_int=1
        else:
            label_int=0
        #print('label:', label, label_int)
        #print(f's1:({s1_from},{s1_to}), s2:({s2_from},{s2_to})') #print('s1:',s1_from, s1_to,', s2:',s2_from, s2_to)
        #print(sent1[s1_from:s1_to], sent2[s2_from:s2_to])#사장 사장 (sent1,2에서 단어위치 출력)

        #get start position's token indexs(for tok1,tok2)
        tokStartIdx, tokEndIdx = [], []
        for tokens, pos_s, pos_e in [(tok1,s1_from,s1_to), (tok2,s2_from,s2_to)]: #tok1,tok2 두 tokenized 쌍에서 token의 start/end index 찾기.
            findToks, findToke = True, True
            token_idx=0 #[CLS] 무시.
            ctr=-1#5 #current token rear #[CLS]=5
            #print(f'tokens: pos_s:{pos_s}, pos_e:{pos_e}')
            for curTok in tokens['input_ids']: #tok2['input_ids']:
                curTok_org = self.tokenizer.decode(curTok) #int -> str
                len_tok=getTokLength(curTok_org.replace(" ",""))#len(curTok_org)
                #print(f'curTok: {curTok_org.replace(" ","")}({curTok}) {ctr}+{len_tok}=',ctr+len_tok)
                if curTok==2: #case1: [CLS]
                    continue
                else: #case3: keep searching word index
                    ctr+=len_tok
                    #token_idx+=1 #move to next token
                
                if findToks and pos_s<=ctr: #case2.1: meet word-token sIndex
                    tokStartIdx.append(token_idx)
                    findToks = False
                    pass
                    #print(f'2append start_tok at {token_idx}, str_idx={ctr}({pos_s})')
                if findToke and ctr>=pos_e: #case2.2: meet word-token eIndex
                    tokEndIdx.append(token_idx) #append token index
                    findToke = False
                    #print(f'append end_tok at {token_idx}, str_idx={ctr}({pos_e})')
                    break


                token_idx+=1 #move to next token
        #print('token StartIdx:',tokStartIdx,' endIdx:',tokEndIdx) #token StartIdx: [1, 1](사장예문) -> 잘 안골라지는데 ctr,tokStartIdx,tokEndIdx,token_idx 뽑으면서 다시보기.(추후에 )

        #tok_pos = [] #하.. 이제 tok에서 문장수준 단어임베딩 찾아서 두개를 유사도 이진분류하면 되는데 안되넹...

        input_ids1=torch.LongTensor(tok1["input_ids"]) #input token index in vacabs
        token_type_ids1=torch.LongTensor(tok1["token_type_ids"]) #segment token index 
        attention_mask1=torch.LongTensor(tok1["attention_mask"]) #boolean: masking attention(0), not masked(1)

        input_ids2=torch.LongTensor(tok2["input_ids"]) #input token index in vacabs
        token_type_ids2=torch.LongTensor(tok2["token_type_ids"]) #segment token index 
        attention_mask2=torch.LongTensor(tok2["attention_mask"]) #boolean: masking attention(0), not masked(1)

        #print(input_ids1.shape, token_type_ids1.shape, attention_mask1.shape)
        #print(input_ids2.shape, token_type_ids2.shape, attention_mask2.shape)
        #print(label_int, tokStartIdx, tokEndIdx)
            
        return input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2, label_int, tokStartIdx[0], tokEndIdx[0], tokStartIdx[1], tokEndIdx[1] #tensor(400), tensor(400), tensor(400), int, [int,int]
        #return type(len): tensor(2*tokMaxLen)*3개, tensor(2*tokMaxLen)*3개,  tensor(int), tensor(int), tensor(int)
        #current len: 200,200,200, 200,200,200, 1, 1, 1


class COPA_biSentence(Dataset): 
    def __init__(self, data_filename): #str: path of csv
        super(COPA_biSentence,self).__init__()
        self.data = pd.read_csv(os.path.join(os.getcwd(),data_filename), sep="\t")#[:100] #data: ~800개까진 OOM 안뜸.
        self.tokenizer = model_tokenizer.from_pretrained(model_path)
    
    def __len__(self): #return size of dataset(row*column)
        return len(self.data)
    
    def __name__(self): #return name
        return 'COPA_biSentence'
    
    def __getitem__(self, item): #fetch data sample(row) for given key(item)
        sentData = self.data.iloc[item,:]
        max_tokenizer_length = TOKEN_MAX_LENGTH['COPA']

        sentence = sentData['sentence'][:max_tokenizer_length] #str
        sent1 = sentData['1'][:max_tokenizer_length] #str
        sent2 = sentData['2'][:max_tokenizer_length] #str
        qType = sentData['question'] #'원인', '결과'

        input1_str, input2_str = '',''
        if qType == '결과':
            input1_str = sentence+self.tokenizer.sep_token+sent1 #[CLS:2]+S+[SEP:3]+S1+[SEP]+[PAD:1]
            input2_str = sentence+self.tokenizer.sep_token+sent2 #[CLS:2]+S+[SEP:3]+S2+[SEP]+[PAD:1]
        elif qType == '원인':
            input1_str = sent1+self.tokenizer.sep_token+sentence #[CLS:2]+S+[SEP:3]+S1+[SEP]+[PAD:1]
            input2_str = sent2+self.tokenizer.sep_token+sentence #[CLS:2]+S+[SEP:3]+S2+[SEP]+[PAD:1]
        else:
            pass
        tok1 = self.tokenizer(input1_str, padding="max_length", max_length=max_tokenizer_length*2, truncation=True)
        tok2 = self.tokenizer(input2_str, padding="max_length", max_length=max_tokenizer_length*2, truncation=True)
        #print('tokenized:',self.tokenizer.tokenize(input_str))
        #print('origin sentence:',self.tokenizer.decode(tok['input_ids']))
        #print('token_set:',tok) #(input_ids, token_type_ids, attention_mask)

        label = sentData['Answer']
        #print(f'sent:{sentence}, sent1:{sent1}, sent2:{sent2}, type:{qType}, label:{label}')
        #print(f'type: sent:{type(sentence)}, sent1:{type(sent1)}, sent2:{type(sent2)}, type:{type(qType)}, label:{type(label)}')
        label_int = 0
        if label==1:
            label_int=0
        elif label==2:
            label_int=1
        else:
            pass
        #print(f'qType:{qType}. tok1:{tok1}, tok2:{tok2}, label_int:{label_int}')
        #print(f'max_len:{max_tokenizer_length}, qType:{type(qType)}. tok1:{type(tok1)}, tok2:{type(tok2)}, label_int:{type(label_int)}')
        
        input_ids1=torch.LongTensor(tok1["input_ids"]) #input token index in vacabs
        token_type_ids1=torch.LongTensor(tok1["token_type_ids"]) #segment token index 
        attention_mask1=torch.LongTensor(tok1["attention_mask"]) #boolean: masking attention(0), not masked(1)

        input_ids2=torch.LongTensor(tok2["input_ids"]) #input token index in vacabs
        token_type_ids2=torch.LongTensor(tok2["token_type_ids"]) #segment token index 
        attention_mask2=torch.LongTensor(tok2["attention_mask"]) #boolean: masking attention(0), not masked(1)
            
        return input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2, label_int 
        #return type(len): tensor(2*tokMaxLen), tensor(2*tokMaxLen), tensor(2*tokMaxLen), tensor(int), tensor(int), tensor(int)
        #current len: 100, 100, 100, 100, 100, 100, 1 

##monologg/KoBERT-nsmc##
class BoolQA_dataset(Dataset): 
    def __init__(self, data_filename): #str: path of csv
        super(BoolQA_dataset,self).__init__()
        self.data = pd.read_csv(os.path.join(os.getcwd(),data_filename), sep="\t")#[:100] #data: ~800개까진 OOM 안뜸.
        self.tokenizer = model_tokenizer.from_pretrained(model_path)
    
    def __len__(self): #return size of dataset(row*column)
        return len(self.data)
    
    def __name__(self): #return name
        return 'BoolQA_dataset'
        
    def __getitem__(self, item): #fetch data sample(row) for given key(item)
        qaData = self.data.iloc[item,:]
        max_text_length, max_question_length = TOKEN_MAX_LENGTH['BoolQ']
        #print(len(sentData), type(sentData), sentData) #4, Seires, (csv: source, accep_label, source_annot, sentence)

        text = qaData['Text'][:max_text_length]
        question = qaData['Question'][:max_question_length]

        input_str = question+self.tokenizer.sep_token+text #[CLS:2]+question+[SEP:3]+text+[SEP]+[PAD:1]
        max_tokenizer_length = max_text_length+max_question_length+5
        tok = self.tokenizer(input_str, padding="max_length", max_length=max_tokenizer_length, truncation=True) #PreTrainedTokenizer.__call__(): str -> tensor(3*400)
        #print(tok) #(input_ids, token_type_ids, attention_mask)
        #[token_ids], [0], [1,1,1,0....]

        #tok_text = self.tokenizer(text, padding="max_length", max_length=max_text_length, truncation=True)
        #tok_question = self.tokenizer(question, padding="max_length", max_length=max_question_length, truncation=True)
        #print('token_text: ',len(tok_text["input_ids"]), tok_text["input_ids"])
        #print('token_ques: ',len(tok_question["input_ids"]), tok_question["input_ids"])
        #tok = tok_text+self.tokenizer.sep_token+tok_question
        #print('tokenTypes: ',self.tokenizer.create_token_type_ids_from_sequences) #UNK, SEP, PAD, CLS, MASK
        clsIdx, sepIdx_s = tok["input_ids"].index(2), tok["input_ids"].index(3)
        sepIdx_e = sepIdx_s+1+tok["input_ids"][sepIdx_s+1:].index(3)
        #print(self.tokenizer.sep_token, f'({tok["input_ids"][clsIdx]})', sepIdx_s, sepIdx_e)
        #print(tok["input_ids"][sepIdx_s-2:sepIdx_s+2], tok["input_ids"][sepIdx_e-2:sepIdx_e+2])
        #print('token: ', len(tok["input_ids"]), tok["input_ids"])

        
        label=qaData['Answer(FALSE = 0, TRUE = 1)'] #pd.read_csv("answersList.csv", sep=",")[:100]
        #print(label)

        #print(f'text:{text}, question:{question}, label:{label}') #
        #(f'type: text:{type(text)}, question:{type(question)}, label:{type(label)}') #text:<class 'str'>, question:<class 'str'>, label:<class 'numpy.int64'>

        #print(f'max_len:{max_tokenizer_length}, tok:{type(tok)}, label:{type(label)}') #max_len:585, tok:<class 'transformers.tokenization_utils_base.BatchEncoding'>, label:<class 'numpy.int64'>
        #print(f'shape_tok:{len(tok)}, tok:{tok["input_ids"]}, label:{label}') #3, 

        input_ids=torch.LongTensor(tok["input_ids"]) #input token index in vacabs
        token_type_ids=torch.LongTensor(tok["token_type_ids"]) #segment token index 
        attention_mask=torch.LongTensor(tok["attention_mask"]) #boolean: masking attention(0), not masked(1)
        #print('dataset shape: ',input_ids.shape, token_type_ids.shape, attention_mask.shape, label) #torch.Size([585]) torch.Size([585]) torch.Size([585]) 0
        #print('dataset type: ', type(input_ids), type(token_type_ids), type(attention_mask), type(label)) #<class 'torch.Tensor'> x 3개,  <class 'numpy.int64'>
            
        return input_ids, token_type_ids, attention_mask, label, sepIdx_s, sepIdx_e #tensor(400), tensor(400), tensor(400), int
        #return type(len): tensor(2*(textMaxlen+quesMaxlen)) x 3개, tensor(int)
        #current len: 485, 485, 485, 1 
