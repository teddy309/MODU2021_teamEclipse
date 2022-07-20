import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import pandas as pd
import numpy as np
import os

from PIL import Image
import copy

from pprint import pprint
from datetime import datetime

#from data.getIndex import getIdxTopN
#from sklearn.metrics import matthews_corrcoef

##
#from transformers import AutoModel, AutoTokenizer, AutoModelWithLMHead #klue/roberta, Beomi/KcBERT
#from transformers import ElectraTokenizer #monologg/koElectra
#from transformers import BertModel #BertTokenizer, BertModel #monologg/kobert
#from tokenization_kobert import KoBertTokenizer #monologg/kobert
#from transformers import T5Model, T5Tokenizer, T5ForConditionalGeneration #kolang-t5-base
from transformers import AdamW

from utils import compute_metrics,  MCC, get_label, set_seed
from utils import MODEL_CLASSES, MODEL_PATH_MAP
from utils import TOKEN_MAX_LENGTH #SPECIAL_TOKENS
from utils import getParentPath, save_model, load_model, save_json, DATASET_PATHS
from kobert_datasets import COLA_dataset
from kobert_models import model_COLA, model_COPA_biSent

data_path=os.getcwd()+'/../../dataset/'
model_name = 'koelectra' #'kobert', 'roberta-base', 'koelectra'
task_name = 'COLA' #'COLA', 'WiC', 'COPA', 'BoolQ'

taskDir_path, fname_train, fname_dev, fname_test, _ = DATASET_PATHS[task_name]

data_path=os.getcwd()+'/../../dataset/'
max_tokenizer_length = TOKEN_MAX_LENGTH[task_name] #100 #20 #WiC 데이터는 (SENTENCE1,SENTENCE2) 문장길이가 길어서 maxlength가 넘으면 잘라줬음. 나중에 넘는건 제외하는 식으로 바꿔야함.


#config_class, model_class, model_tokenizer = MODEL_CLASSES['kobert'] #

countEpoch = 0

bestMCC = -2 #-1 ~ +1
bestAcc = -1 #0 ~ 1
bestLoss = 1 #0 ~ 1
bestLoss_at = 0

def train_cola(model, data_loader, batch_size, epochs, lf, optimizer, device):
    model.train() #set model training mode

    min_loss = 1 #initial value(0~1)
    for _ in range(epochs):
        correct = 0
        all_loss = []
        mini_batch = 0
        print(f'[epoch {countEpoch+_}]') #print(f'[epoch {_}]')
        for batchIdx, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_loader):
            #print('[epoch, batch_index]',_ ,batchIdx) #
            #print('input(iids) shape:',input_ids.shape) #torch.Size([10, 20])

            #print(input_ids.shape, token_type_ids.shape, attention_mask.shape, label)
            #torch.Size([10, 20])x3개, tensor([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])

            #device = torch.device('cuda:0')
            #model.to(device)
            model.zero_grad() #model weight 초기화 맞나?

            input_ids = input_ids.to(device)#.cuda() #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)#.cuda() #
            attention_mask = attention_mask.to(device)#.cuda() #
            label = label.long().to(device)#.cuda()

            #print('model input(iids):',input_ids.shape) #torch.Size([10, 20])
            output = model(input_ids, token_type_ids, attention_mask)
            #print('model output(all):',output.shape) #torch.Size([10, 2])
            #print('label shape: ',label.shape)

            #output=output[:,0,:] #

            lf_input=output#.view_as(label) #output
            lf_target=label#[batchIdx] #label
            #print('loss_input shape: ',lf_input.shape) #torch.Size([10, 2])
            #print('loss_target shape: ',lf_target.shape) #torch.Size([10])

            loss = lf(output,label) #lf(output,label)
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
        accuracy = (correct / len(colaDataset_train)).float()
        print("acc = ", accuracy,", loss = ",avg_loss)
        
        #bestLoss = min(bestLoss, avg_loss)
        min_loss = min(min_loss, avg_loss)

    return min_loss

def eval_cola(model, data_loader, batch_size, device):
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
            
            logits = output#torch.argmax(output,dim=-1)


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

def train_model(model, data_loader_list, batch_size, epochs, lf, optimizer, device):
    TrainLoader, EvalLoader, _ = data_loader_list
    countEpoch, bestLoss_at= 0, 0
    minLoss = 1
    
    for epoch in range(int(epochs/num_printEval)):
        mcc, accuracy = eval_cola(model, EvalLoader, bs, device)
        print(f'before epoch{epoch}: devSet(MCC:{mcc:.4f}, acc:{accuracy:.4f}), testSet(MCC:{mccTest:.4f}, acc:{accTest:.4f})') #
        bestMCC = max(bestMCC,mcc)
        bestAcc = max(bestAcc,accuracy)

        minLoss = train_cola(mymodel, TrainLoader, bs, num_printEval, lf, optimizer, device) #3epoch마다 eval
        if minLoss < bestLoss:
            bestLoss = minLoss
            bestLoss_at = countEpoch
        countEpoch+=num_printEval #countEpoch 업데이트
    return minLoss

##monologg/KoBERT##
if __name__ == "__main__":
    homePth = getParentPath(os.getcwd())
    datasetPth = homePth+'/dataset/'
    print('homePth:',homePth,', curPth:',os.getcwd())
    start_day_time=datetime.now().strftime("%m/%d, %H:%M:%S")
    print('training start at (date, time): ',start_day_time)

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
    model_savePth = homePth+f'/model_pth/{model_name}_{task_name}/'
    model_loadPth = homePth+f'/model_pth/{model_name}_{task_name}/'
    filename_modelSave_chkpt = f'modelpth_{task_name}_{model_type}_epoch{countEpoch}by{num_printEval}_bs{bs}.pt'
    filename_modelLoad_chkpt = f'{task_name}_{model_type}_epoch(50.15)by{num_printEval}_bs{bs}_rs{random_seed_int}.pt' #최대epoch에서의 model path(epochs,epoch) 맞게 입력.

    #f'modelpth_{task_name}_{model_type}_epoch{epochs}by{num_printEval}_bs{bs}.pt' #epoch 다 돌았을때의 model path
    print('save_model:',bool_save_model,'filename_modelSave_chkpt: ',filename_modelSave_chkpt)
    print('load_model:',bool_load_model,'filename_modelLoad_chkpt: ',filename_modelLoad_chkpt)

    lf = nn.CrossEntropyLoss()

    mymodel = model_COLA()
    if not bool_load_model:
        #lf = nn.MSELoss()
        pass
    elif bool_load_model:
        mymodel = load_model(model_loadPth, filename_modelLoad_chkpt, mymodel, device)
    print('mymodel is biBERT: ',hasattr(mymodel,'cosSim'),', load_model: ',bool_load_model) #cosSim유무로 bi/uni model여부 판단. 
    mymodel.to(device)

    optimizer = Adam(mymodel.parameters(),lr=2e-5, eps=1e-8)
    #optimizer = AdamW(mymodel.parameters(), lr=1e-5)

    colaDataset_train = COLA_dataset(os.path.join(os.getcwd(),tsvPth_train)) #(dataPth_dev)
    colaDataset_dev = COLA_dataset(os.path.join(os.getcwd(),tsvPth_dev)) #(dataPth_dev)
    colaDataset_test = COLA_dataset(os.path.join(os.getcwd(),tsvPth_test)) #(dataPth_test)
    
    TrainLoader = DataLoader(colaDataset_train, batch_size=bs)
    EvalLoader = DataLoader(colaDataset_dev, batch_size=bs)
    InferenceLoader = DataLoader(colaDataset_test, batch_size=bs)
    DataLoaders = [TrainLoader, EvalLoader, InferenceLoader]

    print('[Training Phase]')
    print(f'len {task_name}_train:{len(colaDataset_train)}, batch_size:{bs}, epochs:{epochs}(eval_by {num_printEval}), device({device})')

    train_model(mymodel, DataLoaders, bs, epochs, lf, optimizer, device)
    '''
    for epoch in range(int(epochs/num_printEval)):
        result, accuracy = eval_cola(mymodel, EvalLoader, bs, device)
        mccTest, accTest = eval_cola(mymodel, InferenceLoader, bs, device) #
        print(f'before epoch{epoch}: devSet(MCC:{result:.4f}, acc:{accuracy:.4f}), testSet(MCC:{mccTest:.4f}, acc:{accTest:.4f})') #
        bestMCC = max(bestMCC,result)
        bestAcc = max(bestAcc,accuracy)

        minLoss = train_cola(mymodel, TrainLoader, bs, num_printEval, lf, optimizer, device) #3epoch마다 eval
        if minLoss < bestLoss:
            bestLoss = minLoss
            bestLoss_at = countEpoch
        countEpoch+=num_printEval #countEpoch 업데이트

        if bool_save_model:
            filename_modelSave_chkpt = f'{task_name}_{model_type}_epoch{epochs}.{countEpoch}by{num_printEval}_bs{bs}_rs{random_seed_int}.pt'
            save_model(model_savePth, filename_modelSave_chkpt, mymodel, optimizer)
    '''
    
    print('[Evaluation Phase]')
    print(f'len {task_name}_dev:{len(colaDataset_dev)}, batch_size:{bs}, epochs:{epochs}, device({device})')
    result, accuracy = eval_cola(mymodel, EvalLoader, bs, device)
    bestMCC = max(bestMCC,result)
    bestAcc = max(bestAcc,accuracy)
    #print(f'bestMCC:{bestMCC}, bestAccuracy:{bestAcc}, bestLoss:{bestLoss}')

    print('[Inference Phase]')
    #InferenceLoader = DataLoader(colaDataset_test, batch_size=bs)
    eval_cola(mymodel, InferenceLoader, bs, device) #test acc 결과뽑기.
    modelOutput = inference_cola(mymodel, InferenceLoader, bs, device)

    end_day_time=datetime.now().strftime("%m/%d, %H:%M:%S") #Date %m/%d %H:%M:%S
    print(f'training model from {start_day_time} to {end_day_time} (date, time): ')

    ##save model path##
    if bool_save_output and bool_load_model:
        model_outputPth = homePth+'/output/'
        modelOutputPth = filename_modelLoad_chkpt+f'_{model_name}ExTrain_epc{epochs}by{num_printEval}_bs{bs}.json' #json file
        save_json(model_outputPth, modelOutputPth, task_name.lower(), modelOutput)
        #save_model(model_outputPth, modelOutputPth, mymodel, optimizer)
    if bool_load_model:
        pass
    elif bool_save_model:
        day_time=datetime.now().strftime("d%m%d_t%H%M") #date, time
        filename_modelSave_chkpt = f'{day_time}_{task_name}_{model_type}_epoch{countEpoch}by{num_printEval}_bs{bs}.pt'
        save_model(model_savePth, filename_modelSave_chkpt, mymodel, optimizer)
    else:
        pass

    print('finish')
    print('<SUMMARY>')
    print(f'task:{task_name}, model:{model_name}({model_type}), bs:{bs}, epochs:{epochs}, load/save model:{bool_load_model}/{bool_save_model}, randSeedNum:{random_seed_int}')
    if bool_load_model:
        print(f'model chkpoint_loadPth:{model_loadPth}')
    if bool_save_model:
        print(f'model chkpoint_savePth:{model_savePth}')
    if bool_save_output:
        print(f'model test outputPth(json):{model_outputPth}{modelOutputPth}')
    print(f'bestAccuracy:{bestAcc}, bestMCC:{bestMCC}, bestLoss:{bestLoss}(bestLoss around epoch {bestLoss_at})')

    print('end main')



