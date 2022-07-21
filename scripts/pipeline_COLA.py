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

from utils import compute_metrics,  MCC, get_label, set_seed
from utils import MODEL_PATH_MAP
from utils import TOKEN_MAX_LENGTH #SPECIAL_TOKENS
from utils import getParentPath, save_model, load_model, save_json, DATASET_PATHS #print_timeNow


from kobert_datasets import COLA_dataset
from kobert_models import model_COLA

data_path=os.getcwd()+'/../../dataset/'
model_name = 'koelectra' #'kobert', 'roberta-base', 'koelectra'
task_name = 'COLA' #'COLA', 'WiC', 'COPA', 'BoolQ'

taskDir_path, fname_train, fname_dev, fname_test, _ = DATASET_PATHS[task_name]

data_path=os.getcwd()+'/../../dataset/'
max_tokenizer_length = TOKEN_MAX_LENGTH[task_name] #100 #20 #WiC 데이터는 (SENTENCE1,SENTENCE2) 문장길이가 길어서 maxlength가 넘으면 잘라줬음. 나중에 넘는건 제외하는 식으로 바꿔야함.

countEpoch = 0

bestMCC = -2 #-1 ~ +1
bestAcc = -1 #0 ~ 1
bestLoss = 1 #0 ~ 1
bestLoss_at = 0

def train_cola(model, data_loader, batch_size, epochs, lf, optimizer, device):
    model.train() #set model training mode: gradient 업데이트(O)

    min_loss = 1 #initial value(0~1)
    for _ in range(epochs):
        correct = 0
        all_loss = []
        mini_batch = 0
        print(f'[epoch {countEpoch+_}]') #print(f'[epoch {_}]')
        for batchIdx, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_loader):
            #print(input_ids.shape, token_type_ids.shape, attention_mask.shape, label)
            #batch inputs shape: #max_len=64(TOKEN_MAX_LENGTH), bs=20(pipeline_main) 으로 세팅
            #   - torch.Size([max_len, bs])x3개 (input_ids, token_type_ids, attention_mask)
            #   - tensor([0, 1, 0, 0, 1, 1, 0, 1, 1, 0]) (label) ->len=max_len

            model.zero_grad() #model-params optimizer의 gradient를 0으로 초기화.

            #device = torch.device('cuda:0') #device: 'cpu' 'cuda:0' 'cuda:1'
            #model.to(device)

            #move param_buffers from cpu to gpu     (.to(device) == .cuda())
            input_ids = input_ids.to(device) #torch.Size([10, bs])
            token_type_ids = token_type_ids.to(device) #torch.Size([10, bs])
            attention_mask = attention_mask.to(device) #torch.Size([10, bs])
            label = label.long().to(device) #tensor([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])

            output = model(input_ids, token_type_ids, attention_mask) #torch.Size([10, bs]) -> torch.Size([10, 2])

            lf_input=output #torch.Size([10, 2])
            lf_target=label #torch.Size([10])
            loss = lf(output,label) #torch.Size([20, 2]), tensor

            pred = torch.argmax(output,dim=-1) #torch.Size([10,1]) #model-output 중 큰 값의 인덱스 선택(0/1)
            correct += sum(pred.detach().cpu()==label.detach().cpu())
            all_loss.append(loss)
            loss.backward() #기울기 계산
            optimizer.step() #가중치 업데이트
            mini_batch += batch_size
            #print(mini_batch,"/",len(colaDataset)) #batch학습 진행률: batch/전체Dataset 
        
        #print(sum(all_loss)/len(all_loss))
        #print("acc = ", correct / len(colaDataset))
        avg_loss = (sum(all_loss)/len(all_loss)).detach().cpu().float()
        accuracy = (correct / len(colaDataset_train)).float()
        print("acc = ", accuracy,", loss = ",avg_loss)
        
        #bestLoss = min(bestLoss, avg_loss)
        min_loss = min(min_loss, avg_loss)

    return min_loss
#   return min_loss
def eval_cola(model, data_loader, batch_size, device):
    model.eval() #set model eval mode: gradient 업데이트(X)

    y_true = None #label list
    y_pred = None #model prediction list
    for batchIdx, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_loader):
        with torch.no_grad(): #autograd 끔->속도향상. (Option: model.eval()하면 안해줘도 됨.)
            #move param_buffers to gpu
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.long().to(device)

            output = model(input_ids, token_type_ids, attention_mask)
            
            logits = output #torch.argmax(output,dim=-1)

        if y_pred is None:
            y_pred = logits.detach().cpu().numpy()
            y_true = label.detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, logits.detach().cpu().numpy(), axis=0)
            y_true = np.append(
                y_true, label.detach().cpu().numpy(), axis=0)

    y_pred = np.argmax(y_pred, axis=1)
    result = MCC(y_pred, y_true)
    accuracy = compute_metrics(y_pred, y_true)["acc"]
    #print('eval_MCC = ',result)
    #print('eval_acc = ',accuracy)
    print('eval_MCC = ',result,', eval_acc = ',accuracy)

    return result, accuracy
#   return result, accuracy
def inference_cola(model, data_loader, batch_size, device):
    model.eval()

    y_pred = None #model prediction list

    for batchIdx, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_loader):
        with torch.no_grad(): #autograd 끔->속도향상. 사실 model.eval()하면 안해줘도 됨.
            input_ids = input_ids.to(device) #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)

            output = model(input_ids, token_type_ids, attention_mask)
            
            logits = output#torch.argmax(output,dim=-1)

        if y_pred is None:
            y_pred = logits.detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, logits.detach().cpu().numpy(), axis=0)
    y_pred = np.argmax(y_pred, axis=1)

    print('output shape: ',y_pred.shape, type(y_pred))

    return y_pred
#   return y_pred

##monologg/KoBERT##
if __name__ == "__main__":
    homePth = getParentPath(os.getcwd())
    datasetPth = homePth+'/dataset/'
    print('homePth:',homePth,', curPth:',os.getcwd())
    #start_day_time=print_timeNow()
    #print('training start at (date, time): ',print_timeNow())

    tsvPth_train = datasetPth+taskDir_path+fname_train  #'task1_grammar/NIKL_CoLA_train.tsv'
    tsvPth_dev = datasetPth+taskDir_path+fname_dev #'task1_grammar/NIKL_CoLA_dev.tsv'
    tsvPth_test = datasetPth+taskDir_path+fname_test #'task1_grammar/NIKL_CoLA_test.tsv' 

    bs = 400 #gpu 가능한 bs: [10,20,100,200,400]
    epochs= 0 #50#2#100 #시도한 epochs: [10, 100]
    num_printEval = 1 #4 #꼭 epochs의 약수가 되게 넣어주기. 안그럼 epochs가 조금 모자라게 돔.

    device = torch.device('cuda:1')
    model_type = 'uniBert' # 'uniBert', 'biBERT'
    random_seed_int = 5 # 랜덤시드 넘버=5 로 고정
    set_seed(random_seed_int, device) #random seed 정수로 고정.

    bool_save_model, bool_load_model, bool_save_output = False, False, False #default: True, False, False
    ## model save/load path & save_output_path ##

    lf = nn.CrossEntropyLoss()

    mymodel = model_COLA()
    mymodel.to(device)

    optimizer = Adam(mymodel.parameters(),lr=2e-5, eps=1e-8)
    #optimizer = AdamW(mymodel.parameters(), lr=1e-5)

    colaDataset_train = COLA_dataset(os.path.join(os.getcwd(),tsvPth_train)) #(dataPth_dev)
    colaDataset_dev = COLA_dataset(os.path.join(os.getcwd(),tsvPth_dev)) #(dataPth_dev)
    colaDataset_test = COLA_dataset(os.path.join(os.getcwd(),tsvPth_test)) #(dataPth_test)
    
    TrainLoader = DataLoader(colaDataset_train, batch_size=bs)
    EvalLoader = DataLoader(colaDataset_dev, batch_size=bs)
    InferenceLoader = DataLoader(colaDataset_test, batch_size=bs)

    print('[Training Phase]')
    print(f'len {task_name}_train:{len(colaDataset_train)}, batch_size:{bs}, epochs:{epochs}(eval_by {num_printEval}), device({device})')
    for epoch in range(int(epochs/num_printEval)):
        result, accuracy = eval_cola(mymodel, EvalLoader, bs, device)
        print(f'before epoch{epoch}: devSet(MCC:{result:.4f}, acc:{accuracy:.4f})') #
        #mccTest, accTest = eval_cola(mymodel, InferenceLoader, bs, device) #
        #print(f'before epoch{epoch}: testSet(MCC:{mccTest:.4f}, acc:{accTest:.4f})') #
        bestMCC = max(bestMCC,result)
        bestAcc = max(bestAcc,accuracy)

        minLoss = train_cola(mymodel, TrainLoader, bs, num_printEval, lf, optimizer, device) #3epoch마다 eval
        if minLoss < bestLoss:
            bestLoss = minLoss
            bestLoss_at = countEpoch
        countEpoch+=num_printEval #countEpoch 업데이트

    print('[Evaluation Phase]')
    print(f'len {task_name}_dev:{len(colaDataset_dev)}, batch_size:{bs}, epochs:{epochs}, device({device})')
    result, accuracy = eval_cola(mymodel, EvalLoader, bs, device)
    bestMCC = max(bestMCC,result)
    bestAcc = max(bestAcc,accuracy)
    #print(f'Dev - bestMCC:{bestMCC}, bestAccuracy:{bestAcc}, bestLoss:{bestLoss}')

    print('[Inference Phase]')
    eval_cola(mymodel, InferenceLoader, bs, device) #test acc 결과뽑기.
    modelOutput = inference_cola(mymodel, InferenceLoader, bs, device)

    ## TODO: save model path ##

    #end_day_time=print_timeNow()
    #print(f'training model from {start_day_time} to {end_day_time} (date, time): ')
    print('finish')
    print('<SUMMARY>')
    print(f'task:{task_name}, model:{model_name}({model_type}), bs:{bs}, epochs:{epochs}, load/save model:{bool_load_model}/{bool_save_model}, randSeedNum:{random_seed_int}')
    print(f'bestAccuracy:{bestAcc}, bestMCC:{bestMCC}, bestLoss:{bestLoss}(bestLoss around epoch {bestLoss_at})')

    print('end main')



