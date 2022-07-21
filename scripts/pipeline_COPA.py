import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import pandas as pd
import numpy as np
import os

from pprint import pprint
from datetime import datetime

from utils import compute_metrics, get_label, set_seed
from utils import MODEL_PATH_MAP
from utils import TOKEN_MAX_LENGTH #SPECIAL_TOKENS
from utils import getParentPath, save_model, load_model, save_json, DATASET_PATHS #print_timeNow

from kobert_datasets import COPA_biSentence #COPA_uniSentence, COPA_biSentence
from kobert_models import model_COPA_biSent #model_COPA_uniSent, model_COPA_biSent 

model_name = 'koelectra' #'kobert', 'roberta-base', 'koelectra', 'koelectra_tunib'
task_name = 'COPA' #'COLA', 'WiC', 'COPA', 'BoolQ'
taskDir_path, fname_train, fname_dev, fname_test, _ = DATASET_PATHS[task_name]


data_path=os.getcwd()+'/../../dataset/'
max_tokenizer_length = TOKEN_MAX_LENGTH[task_name] #100 #20 #WiC 데이터는 (SENTENCE1,SENTENCE2) 문장길이가 길어서 maxlength가 넘으면 잘라줬음. 나중에 넘는건 제외하는 식으로 바꿔야함.

## model_WiC_uniSent, model_WiC_biSent : 모델1개에 seq 2개가 들어가는 uni-Bert 구조
## WiC_biSentence, WiC_biSentence : 모델 2개에 seq 하나씩 각각 들어가는 Siamese-BERT 구조.

countEpoch = 0

bestAcc = -1 #0 ~ 1
bestLoss = 1 #0 ~ 1
bestLoss_at = 0


def train_copa_biModel(model, data_loader, batch_size, epochs, lf, optimizer, device):
    model.train() #set model training mode
    min_loss = 1
    for _ in range(epochs):
        correct = 0
        all_loss = []
        mini_batch = 0
        print(f'[epoch {countEpoch+_}]')
        
        #Siamese-BERT: WiC_biSentence(dataset), model_WiC_biSent(model)
        for batchIdx, (input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2, label) in enumerate(data_loader):
            #batch inputs shape: #max_len=40(TOKEN_MAX_LENGTH), bs=20(pipeline_main) 으로 세팅
            #   - torch.Size([max_len, bs])x6개 (input_ids, token_type_ids, attention_mask)x2
            #   - tensor([0, 1, 0, 0, 1, 1, 0, 1, 1, 0]) (label) ->len=max_len

            model.zero_grad() #model weight 초기화

            #move param_buffers from cpu to gpu     (.to(device) == .cuda())
            #Sent1_token
            input_ids1 = input_ids1.to(device) #move param_buffers to gpu
            token_type_ids1 = token_type_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            #Sent2_token
            input_ids2 = input_ids2.to(device)
            token_type_ids2 = token_type_ids2.to(device)
            attention_mask2 = attention_mask2.to(device)
            #label
            label = label.long().to(device)

            output1 = model(input_ids1, token_type_ids1, attention_mask1) #torch.Size([max_len, bs])*3 -> torch.Size([1, bs])
            output2 = model(input_ids2, token_type_ids2, attention_mask2) #torch.Size([max_len, bs])*3 -> torch.Size([1, bs])
            output = torch.cat([output1, output2], dim=1) #(bs,1)*2 -> (bs*2)
            #print('Model output shape: ',output.shape, label.shape, lf) #torch.Size([20, 2]) torch.Size([20]) CrossEntropyLoss()
            
            loss = lf(output,label) #torch.Size([20, 2])

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
        accuracy = (correct / len(copaDataset_train)).float()
        print("acc = ", accuracy,", loss = ",avg_loss)

        min_loss = min(min_loss, avg_loss)

    return min_loss

def eval_copa_biModel(model, data_loader, batch_size, device):
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

            #print(input_ids1.shape, token_type_ids1.shape, attention_mask1.shape) #torch.Size([bs, 100]) * 3
            #print(input_ids2.shape, token_type_ids2.shape, attention_mask2.shape) #torch.Size([bs, 100]) * 3

            output1 = model(input_ids1, token_type_ids1, attention_mask1)
            output2 = model(input_ids2, token_type_ids2, attention_mask2)
            #print('loss_input shape: ',output.shape) #torch.Size([10, 2]) #batch마다 한번씩.
            #print('model output: ',output, output.shape,' -> ', torch.argmax(output,dim=1).shape)
            output = torch.cat([output1, output2], dim=1) #(bs,1)*2 -> (bs*2)
            #print('output_cat shape: ',output.shape, output[0])

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

def inference_copa_biModel(model, data_loader, batch_size, device):
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
    #os.system('ls '+getParentPath(os.getcwd()))
    homePth = getParentPath(os.getcwd())
    datasetPth = homePth+'/dataset/'
    print('homePth:',homePth,', curPth:',os.getcwd())
    #start_day_time=print_timeNow()
    #print('training start at (date, time): ',print_timeNow())
    
    tsvPth_train = datasetPth+taskDir_path+fname_train 
    tsvPth_dev = datasetPth+taskDir_path+fname_dev 
    tsvPth_test = datasetPth+taskDir_path+_#fname_test 

    bs = 20 #100 #10,20,100,200
    epochs= 0 #50 #100 #10
    num_printEval = 1#2 #꼭 epochs의 약수가 되게 넣어주기. 안그럼 epochs가 조금 모자라게 돔.

    device = torch.device('cuda:1')
    model_type = 'biBERT' # 'uniBert', 'biBERT'
    random_seed_int = 5 # 랜덤시드 넘버=5 로 고정
    set_seed(random_seed_int, device) #random seed 정수로 고정.

    bool_save_model, bool_load_model, bool_save_output = False, False, False #default: True, False, False
    ## model save/load path & save_output_path ##

    lf = nn.CrossEntropyLoss()

    mymodel = model_COPA_biSent() #default: bi-BERT(Siamese-BERT)
    mymodel.to(device)

    optimizer = Adam(mymodel.parameters(),lr=2e-5, eps=1e-8)
    #optimizer = AdamW(mymodel.parameters(), lr=1e-5)

    #uni-Bert: WiC_uniSentence(dataset), model_WiC_uniSent(model)
    #Siamese-BERT: WiC_biSentence(dataset), model_WiC_biSent(model)
    #COPA_uniSentence, COPA_biSentence
    copaDataset_train = COPA_biSentence(os.path.join(os.getcwd(),tsvPth_train)) #dataPth_train(default: bi-BERT)
    copaDataset_dev = COPA_biSentence(os.path.join(os.getcwd(),tsvPth_dev)) #dataPth_dev(default: bi-BERT)
    copaDataset_test = COPA_biSentence(os.path.join(os.getcwd(),tsvPth_test)) #dataPth_dev(default: bi-BERT)

    assert model_type == 'biBERT'

    TrainLoader = DataLoader(copaDataset_train, batch_size=bs)
    EvalLoader = DataLoader(copaDataset_dev, batch_size=bs)
    InferenceLoader = DataLoader(copaDataset_test, batch_size=bs)
    print('copaDataset_train is biBERT dataset: ',copaDataset_train.__name__=='COPA_biSentence',', fileName:',tsvPth_train) #__name__ 값으로 bi/uni model 여부 판단
    print('copaDataset_dev is biBERT dataset: ',copaDataset_dev.__name__=='COPA_biSentence',', fileName:',tsvPth_dev) #__name__ 값으로 bi/uni model 여부 판단

    print('[Training Phase]')
    print(f'len {task_name}_train:{len(copaDataset_train)}, batch_size:{bs}, epochs:{epochs}(eval_by {num_printEval}), model:{model_type}, device({device})')
    for epoch in range(int(epochs/num_printEval)):
        accuracy = eval_copa_biModel(mymodel, EvalLoader, bs, device)
        bestAcc = max(bestAcc,accuracy)

        minLoss = train_copa_biModel(mymodel, TrainLoader, bs, num_printEval, lf, optimizer, device) #3epoch마다 eval
        if minLoss < bestLoss:
            bestLoss = minLoss
            bestLoss_at = countEpoch
        countEpoch+=num_printEval #countEpoch 업데이트

    print('[Evaluation Phase]')
    print(f'len {task_name}_dev:{len(copaDataset_dev)}, batch_size:{bs}, epochs:{epochs}, model:{model_type}, device({device})')
    result = eval_copa_biModel(mymodel, EvalLoader, bs, device)
    bestAcc = max(bestAcc,result)
    #print(f'Dev - bestAccuracy:{bestAcc}, bestLoss:{bestLoss}')

    print('[Inference Phase]')
    eval_copa_biModel(mymodel, InferenceLoader, bs, device) #test acc 결과뽑기.
    modelOutput = inference_copa_biModel(mymodel, InferenceLoader, bs, device)

    ## TODO: save model path ##

    #end_day_time=print_timeNow()
    #print(f'training model from {start_day_time} to {end_day_time} (date, time): ')
    print('finish')
    print('<SUMMARY>')
    print(f'task:{task_name}, model:{model_name}({model_type}), bs:{bs}, epochs:{epochs}, load/save model:{bool_load_model}/{bool_save_model}, randSeedNum:{random_seed_int}')
    print(f'bestAccuracy:{bestAcc}, bestLoss:{bestLoss}(bestLoss around epoch {bestLoss_at})')




