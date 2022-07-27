import torch
from torch import nn
import torch.nn.functional as F #cosLoss
#from torch.nn.modules.loss import SmoothL1Loss
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import pandas as pd
import numpy as np
import os

from pprint import pprint
from datetime import datetime

from transformers import BertModel #BertTokenizer, BertModel #monologg/kobert
from tokenization_kobert import KoBertTokenizer #monologg/kobert
#from transformers import T5Model, T5Tokenizer, T5ForConditionalGeneration #kolang-t5-base

from utils import compute_metrics,  MCC, get_label, set_seed
from utils import MODEL_PATH_MAP
from utils import TOKEN_MAX_LENGTH #SPECIAL_TOKENS
from utils import getParentPath, save_model, load_model, save_json, DATASET_PATHS #print_timeNow

from kobert_datasets import WiC_biSentence #WiC_biSentence_abandoned
from kobert_models import model_WiC_biSent #model_WiC_biSent_abandoned

data_path=os.getcwd()+'/../../dataset/'
max_tokenizer_length = 100 #20 #WiC 데이터는 (SENTENCE1,SENTENCE2) 문장길이가 길어서 maxlength가 넘으면 잘라줬음. 나중에 넘는건 제외하는 식으로 바꿔야함.

model_name = 'koelectra' #'kobert', 'roberta-base', 'koelectra', 'koelectra_tunib'
task_name = 'WiC' #'COLA', 'WiC', 'COPA', 'BoolQ'
taskDir_path, fname_train, fname_dev, fname_test, _ = DATASET_PATHS[task_name]


## model_WiC_uniSent, model_WiC_biSent : 모델1개에 seq 2개가 들어가는 uni-Bert 구조
## WiC_biSentence, WiC_biSentence : 모델 2개에 seq 하나씩 각각 들어가는 Siamese-BERT 구조.

countEpoch = 0

bestAcc = -1 #0 ~ 1
bestLoss = 1 #0 ~ 1
bestLoss_at = 0

def train_wic_biModel(model, data_loader, eval_loader, batch_size, epochs, lf, optimizer, device):
    model.train() #set model training mode
    acc_list = [] #store mini_batch eval accuracy
    min_loss = 1
    for _ in range(epochs):
        correct = 0
        all_loss = []
        mini_batch = 0
        print(f'[epoch {countEpoch+_}]')
        
        #Siamese-BERT: WiC_biSentence(dataset), model_WiC_biSent(model)
        for batchIdx, (input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2, label) in enumerate(data_loader):
            #print('[epoch, batch_index]',_ ,batchIdx) #

            model.zero_grad() #model weight 초기화

            #Sent1_token
            input_ids1 = input_ids1.to(device) #move param_buffers to gpu
            token_type_ids1 = token_type_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            #Sent2_token
            input_ids2 = input_ids2.to(device)
            token_type_ids2 = token_type_ids2.to(device)
            attention_mask2 = attention_mask2.to(device)

            label = label.long().to(device)

            output1 = model(input_ids1, token_type_ids1, attention_mask1) #shape: [bs,768]
            output2 = model(input_ids2, token_type_ids2, attention_mask2) #shape: [bs,768]

            output = torch.cat([output1, output2], dim=1)

            loss = lf(output,label) #torch.Size([20, 2])

            pred = torch.argmax(output,dim=-1)

            correct += sum(pred.detach().cpu()==label.detach().cpu())
            all_loss.append(loss)
            loss.backward() #기울기 계산
            optimizer.step() #가중치 업데이트
            mini_batch += batch_size

            #if mini_batch%1000 == 0:
            ##    print(f'batch{mini_batch}:',end='')
            #    accuracy = eval_wic_biModel(mymodel, eval_loader, bs, device)
            #    acc_list.append(accuracy)
            #    model.train() #set model training mode
            #print(mini_batch,"/",len(colaDataset)) #전체 Dataset 중 batch 단위로 수행완료. 

        #print(sum(all_loss)/len(all_loss))
        #print("acc = ", correct / len(colaDataset))
        avg_loss = (sum(all_loss)/len(all_loss)).detach().cpu().float()
        accuracy = (correct / len(wicDataset_train)).float()
        print("acc = ", accuracy,", loss = ",avg_loss)

        min_loss = min(min_loss, avg_loss)

    return min_loss

def eval_wic_biModel(model, data_loader, batch_size, device):
    model.eval()

    y_true = None #label list
    y_pred = None #model prediction list

    for batchIdx, (input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2, label) in enumerate(data_loader):
        with torch.no_grad(): #autograd 끔->속도향상. 사실 model.eval()하면 안해줘도 됨.
            input_ids1 = input_ids1.to(device) #move param_buffers to gpu
            token_type_ids1 = token_type_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            input_ids2 = input_ids2.to(device)
            token_type_ids2 = token_type_ids2.to(device)
            attention_mask2 = attention_mask2.to(device)
            
            label = label.long().to(device)

            output1 = model(input_ids1, token_type_ids1, attention_mask1)
            output2 = model(input_ids2, token_type_ids2, attention_mask2)

            output = torch.cat([output1, output2], dim=1) #(bs,1)*2 -> (bs*2)

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

def inference_wic_biModel(model, data_loader, batch_size, device):
    model.eval()

    y_pred = None #model prediction list

    for batchIdx, (input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2, label) in enumerate(data_loader):
        with torch.no_grad(): #autograd 끔->속도향상. 사실 model.eval()하면 안해줘도 됨.
            input_ids1 = input_ids1.to(device) #move param_buffers to gpu
            token_type_ids1 = token_type_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            
            input_ids2 = input_ids2.to(device)
            token_type_ids2 = token_type_ids2.to(device)
            attention_mask2 = attention_mask2.to(device)
            
            label = label.long().to(device)

            output1 = model(input_ids1, token_type_ids1, attention_mask1)
            output2 = model(input_ids2, token_type_ids2, attention_mask2)
            output = torch.cat([output1, output2], dim=1) #(bs,1)*2 -> (bs*2)

        if y_pred is None:
            y_pred = output.detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, output.detach().cpu().numpy(), axis=0)
    y_pred = np.argmax(y_pred, axis=1)
    print('output shape: ',y_pred.shape, type(y_pred))

    return y_pred


##monologg/KoBERT##
if __name__ == "__main__":
    homePth = getParentPath(os.getcwd())
    datasetPth = homePth+'/dataset/'
    print('homePth:',homePth,', curPth:',os.getcwd())
    #start_day_time=print_timeNow()
    #print('training start at (date, time): ',print_timeNow())
    
    tsvPth_train = datasetPth+taskDir_path+fname_train #'task2_homonym/NIKL_SKT_WiC_Train.tsv'
    tsvPth_dev = datasetPth+taskDir_path+fname_dev #'task2_homonym/NIKL_SKT_WiC_Dev.tsv'
    tsvPth_test = datasetPth+taskDir_path+fname_test #'task2_homonym/NIKL_SKT_WiC_Test_labeled.tsv'

    bs = 50#50 #100 #10,20,100,200
    epochs= 0#100 #10
    num_printEval = 1#2 #꼭 epochs의 약수가 되게 넣어주기. 안그럼 epochs가 조금 모자라게 돔.

    device = torch.device('cuda:0')
    model_type = 'biBERT' # 'uniBert', 'biBERT'
    random_seed_int = 5 # 랜덤시드 넘버=5 로 고정
    set_seed(random_seed_int, device) #random seed 정수로 고정.

    bool_save_model, bool_load_model, bool_save_output = False, False, False #default: True, False, False
    ## TODO: model save/load path & save_output_path ##

    lf = nn.CrossEntropyLoss()

    assert model_type=='biBERT'
    mymodel = model_WiC_biSent() #default: bi-BERT
    mymodel.to(device)

    optimizer = Adam(mymodel.parameters(),lr=2e-5, eps=1e-8)

    #uni-Bert: WiC_uniSentence(dataset), model_WiC_uniSent(model)
    #Siamese-BERT: WiC_biSentence(dataset), model_WiC_biSent(model)
    wicDataset_train = WiC_biSentence(os.path.join(os.getcwd(),tsvPth_train)) #dataPth_train(default: bi-BERT)
    wicDataset_dev = WiC_biSentence(os.path.join(os.getcwd(),tsvPth_dev)) #dataPth_dev(default: bi-BERT)
    wicDataset_test = WiC_biSentence(os.path.join(os.getcwd(),tsvPth_test)) #dataPth_test(default: bi-BERT)
    
    TrainLoader = DataLoader(wicDataset_train, batch_size=bs)
    EvalLoader = DataLoader(wicDataset_dev, batch_size=bs)
    InferenceLoader = DataLoader(wicDataset_test, batch_size=bs)

    print('[Training Phase]')
    print(f'len WiC_train:{len(wicDataset_train)}, batch_size:{bs}, epochs:{epochs}(eval_by {num_printEval}), model:{model_type}, device({device})')
    for epoch in range(int(epochs/num_printEval)):
        accuracy = eval_wic_biModel(mymodel, EvalLoader, bs, device)
        bestAcc = max(bestAcc,accuracy)

        minLoss = train_wic_biModel(mymodel, TrainLoader, bs, num_printEval, lf, optimizer, device) #3epoch마다 eval
        if minLoss < bestLoss:
            bestLoss = minLoss
            bestLoss_at = countEpoch
        countEpoch+=num_printEval #countEpoch 업데이트

    print('[Evaluation Phase]')
    print(f'len WiC_dev:{len(wicDataset_dev)}, batch_size:{bs}, epochs:{epochs}, model:{model_type}, device({device})')
    result = eval_wic_biModel(mymodel, EvalLoader, bs, device)
    bestAcc = max(bestAcc,result)

    print('[Inference Phase]')
    print(f'len {task_name}_test:{len(wicDataset_dev)}, batch_size:{bs}, epochs:{epochs}, model:{model_type}, device({device})')
    eval_wic_biModel(mymodel, InferenceLoader, bs, device)
    modelOutput = inference_wic_biModel(mymodel, InferenceLoader, bs, device)

    ## TODO: save model path ##

    #end_day_time=print_timeNow()
    #print(f'training model from {start_day_time} to {end_day_time} (date, time): ')
    print('finish')
    print('<SUMMARY>')
    print(f'task:{task_name}, model:{model_name}({model_type}), bs:{bs}, epochs:{epochs}, load/save model:{bool_load_model}/{bool_save_model}, randSeedNum:{random_seed_int}')
    print(f'bestAccuracy:{bestAcc}, bestLoss:{bestLoss}(bestLoss around epoch {bestLoss_at})')




