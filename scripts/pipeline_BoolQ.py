import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import pandas as pd
import numpy as np
import os

from pprint import pprint
from datetime import datetime

from transformers import AdamW


from utils import compute_metrics, get_label, set_seed
from utils import MODEL_CLASSES, MODEL_PATH_MAP
from utils import TOKEN_MAX_LENGTH #SPECIAL_TOKENS
from utils import getParentPath, save_model, load_model, save_json, DATASET_PATHS #print_timeNow

from kobert_datasets import BoolQA_dataset 
from kobert_models import model_BoolQA 

model_name = 'koelectra' #'kobert', 'roberta-base', 'koelectra', 'koelectra_QA', 'koelectra_tunib'
task_name = 'BoolQ' #'COLA', 'WiC', 'COPA', 'BoolQ'
#config_class, model_class, model_tokenizer = MODEL_CLASSES[model_name] #
taskDir_path, fname_train, fname_dev, fname_test, _ = DATASET_PATHS[task_name]

data_path=os.getcwd()+'/../../dataset/'
max_tokenizer_length = TOKEN_MAX_LENGTH[task_name] #100 #20 #WiC 데이터는 (SENTENCE1,SENTENCE2) 문장길이가 길어서 maxlength가 넘으면 잘라줬음. 나중에 넘는건 제외하는 식으로 바꿔야함.

## model_WiC_uniSent, model_WiC_biSent : 모델1개에 seq 2개가 들어가는 uni-Bert 구조
## WiC_biSentence, WiC_biSentence : 모델 2개에 seq 하나씩 각각 들어가는 Siamese-BERT 구조.

countEpoch = 0

#bestMCC = -2 #-1 ~ +1
bestAcc = -1 #0 ~ 1
bestLoss = 1 #0 ~ 1
bestLoss_at = 0


def train_boolq(model, data_loader, batch_size, epochs, lf, optimizer, device):
    model.train() #set model training mode
    min_loss = 1
    for _i in range(epochs):
        correct = 0
        all_loss = []
        mini_batch = 0
        print(f'[epoch {countEpoch+_i}]')
        
        for batchIdx, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_loader):
            model.zero_grad() #model weight 초기화

            #QAset_token
            input_ids = input_ids.to(device) #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)

            label = label.long().to(device)

            output = model(input_ids, token_type_ids, attention_mask) #shape: 
            
            #customLoss = 
            loss = lf(output,label) #lf(output,label)
            #pred = output #torch.argmax(output,dim=-1)
            pred = torch.argmax(output,dim=-1)

            correct += sum(pred.detach().cpu()==label.detach().cpu())
            all_loss.append(loss)
            loss.backward() #기울기 계산
            optimizer.step() #가중치 업데이트
            mini_batch += batch_size
            #print(mini_batch,"/",len(colaDataset)) #전체 Dataset 중 batch 단위로 수행완료. 

        #print(sum(all_loss)/len(all_loss))
        #print("acc = ", correct / len(colaDataset))
        avg_loss = (sum(all_loss)/len(all_loss)).detach().cpu().float()
        accuracy = (correct / len(boolqDataset_train)).float()
        print("acc = ", accuracy,", loss = ",avg_loss)

        min_loss = min(min_loss, avg_loss)

    return min_loss

def eval_boolq(model, data_loader, batch_size, device):
    model.eval()

    y_true = None #label list
    y_pred = None #model prediction list

    for batchIdx, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_loader):
        with torch.no_grad(): #autograd 끔->속도향상. 사실 model.eval()하면 안해줘도 됨.
            input_ids = input_ids.to(device) #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            label = label.long().to(device)

            output = model(input_ids, token_type_ids, attention_mask)

        if y_pred is None:
            y_pred = output.detach().cpu().numpy()
            y_true = label.detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, output.detach().cpu().numpy(), axis=0)
            y_true = np.append(y_true, label.detach().cpu().numpy(), axis=0)

    y_pred = np.argmax(y_pred, axis=1)
    result = compute_metrics(y_pred, y_true)["acc"]
    print('eval_acc = ',result)

    return result

def inference_boolq(model, data_loader, batch_size, device):
    model.eval()

    y_pred = None #model prediction list

    for batchIdx, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_loader):
        with torch.no_grad(): #autograd 끔->속도향상. 사실 model.eval()하면 안해줘도 됨.
            input_ids = input_ids.to(device) #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)

            label = label.long().to(device)

            output = model(input_ids, token_type_ids, attention_mask)

        if y_pred is None:
            y_pred = output.detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, output.detach().cpu().numpy(), axis=0)

    y_pred = np.argmax(y_pred, axis=1)
    #result = compute_metrics(y_pred, y_true)["acc"]
    #print('eval_acc = ',result)
    print('output shape: ',y_pred.shape, type(y_pred))

    return y_pred

##monologg/KoBERT##
if __name__ == "__main__":
    #os.system('ls '+getParentPath(os.getcwd()))
    homePth = getParentPath(os.getcwd())
    datasetPth = homePth+'/dataset/'
    print('homePth:',homePth,', curPth:',os.getcwd())
    #start_day_time=print_timeNow()
    #print('training start at (date, time): ',print_timeNow())
    
    tsvPth_train = datasetPth+taskDir_path+fname_train #'task2_homonym/NIKL_SKT_WiC_Train.tsv'
    tsvPth_dev = datasetPth+taskDir_path+fname_dev #'task2_homonym/NIKL_SKT_WiC_Test_labeled.tsv'
    tsvPth_test = datasetPth+taskDir_path+fname_test #'task2_homonym/NIKL_SKT_WiC_Test_labeled.tsv'

    bs = 20 #100 #10,20,100,200
    epochs= 0 #2#20#0#50#100 #10
    num_printEval = 1#2 #꼭 epochs의 약수가 되게 넣어주기. 안그럼 epochs가 조금 모자라게 돔.

    device = torch.device('cuda:0')
    model_type = 'uniBert' # 'uniBert', 'biBERT'
    random_seed_int = 5 # 랜덤시드 넘버=5 로 고정
    set_seed(random_seed_int, device) #random seed 정수로 고정.

    bool_save_model, bool_load_model, bool_save_output = False, False, False #default: False, True, True
    ## TODO: model save/load path & save_output_path ##

    lf = nn.CrossEntropyLoss()

    mymodel = model_BoolQA() #default: bi-BERT #model_COPA_uniSent, model_COPA_biSent 
    mymodel.to(device)

    #optimizer = Adam(mymodel.parameters(),lr=2e-5, eps=1e-8)
    optimizer = AdamW(mymodel.parameters(), lr=1e-5)

    boolqDataset_train = BoolQA_dataset(os.path.join(os.getcwd(),tsvPth_train)) #(dataPth_train)
    boolqDataset_dev = BoolQA_dataset(os.path.join(os.getcwd(),tsvPth_dev)) #(dataPth_dev)
    boolqDataset_test = BoolQA_dataset(os.path.join(os.getcwd(),tsvPth_test)) #(dataPth_test)
    
    TrainLoader = DataLoader(boolqDataset_train, batch_size=bs)
    EvalLoader = DataLoader(boolqDataset_dev, batch_size=bs)
    InferenceLoader = DataLoader(boolqDataset_test, batch_size=bs)
    print('boolqDataset_train is biBERT dataset: ',boolqDataset_train.__name__=='BoolQA_biSentence',', fileName:',tsvPth_train) #__name__ 값으로 bi/uni model 여부 판단
    print('boolqDataset_dev is biBERT dataset: ',boolqDataset_dev.__name__=='BoolQA_biSentence',', fileName:',tsvPth_dev) #__name__ 값으로 bi/uni model 여부 판단

    print('[Training Phase]')
    print(f'len {task_name}_train:{len(boolqDataset_train)}, batch_size:{bs}, epochs:{epochs}(eval_by {num_printEval}), model:{model_type}, device({device})')
    for epoch in range(int(epochs/num_printEval)):
        accuracy = eval_boolq(mymodel, EvalLoader, bs, device)
        bestAcc = max(bestAcc,accuracy)

        minLoss = train_boolq(mymodel, TrainLoader, bs, num_printEval, lf, optimizer, device) #3epoch마다 eval
        if minLoss < bestLoss:
            bestLoss = minLoss
            bestLoss_at = countEpoch
        countEpoch+=num_printEval #countEpoch 업데이트

    print('[Evaluation Phase]')
    print(f'len {task_name}_dev:{len(boolqDataset_dev)}, batch_size:{bs}, epochs:{epochs}, model:{model_type}, device({device})')
    result = eval_boolq(mymodel, EvalLoader, bs, device)
    bestAcc = max(bestAcc,result)
    #print(f'Dev - bestAccuracy:{bestAcc}, bestLoss:{bestLoss}')

    print('[Inference Phase]')
    eval_boolq(mymodel, InferenceLoader, bs, device) #test acc 결과뽑기.
    modelOutput = inference_boolq(mymodel, InferenceLoader, bs, device)

    ## TODO: save model path ##

    #end_day_time=print_timeNow()
    #print(f'training model from {start_day_time} to {end_day_time} (date, time): ')
    print('<SUMMARY>')
    print(f'task:{task_name}, model:{model_name}({model_type}), bs:{bs}, epochs:{epochs}, load/save model:{bool_load_model}/{bool_save_model}, randSeedNum:{random_seed_int}')
    print(f'bestAccuracy:{bestAcc}, bestLoss:{bestLoss}(bestLoss around epoch {bestLoss_at})')
    
    print('finish')





