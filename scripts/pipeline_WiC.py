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

##from utils import compute_metrics, MCC, MODEL_CLASSES, getParentPath, save_model, load_model #get_label, MODEL_CLASSES, SPECIAL_TOKENS
from utils import compute_metrics,  MCC, get_label, set_seed
from utils import MODEL_CLASSES, MODEL_PATH_MAP
from utils import TOKEN_MAX_LENGTH #SPECIAL_TOKENS
from utils import getParentPath, save_model, load_model, save_json, DATASET_PATHS

from kobert_datasets import WiC_uniSentence, WiC_biSentence
from kobert_models import model_WiC_uniSent, model_WiC_biSent, model_WiC_biSent1

data_path=os.getcwd()+'/../../dataset/'
max_tokenizer_length = 100 #20 #WiC 데이터는 (SENTENCE1,SENTENCE2) 문장길이가 길어서 maxlength가 넘으면 잘라줬음. 나중에 넘는건 제외하는 식으로 바꿔야함.

model_name = 'koelectra' #'kobert', 'roberta-base', 'koelectra', 'koelectra_tunib'
task_name = 'WiC' #'COLA', 'WiC', 'COPA', 'BoolQ'
taskDir_path, fname_train, fname_dev, fname_test, _ = DATASET_PATHS[task_name]

#config_class, model_class, model_tokenizer = MODEL_CLASSES[model_name] #


## model_WiC_uniSent, model_WiC_biSent : 모델1개에 seq 2개가 들어가는 uni-Bert 구조
## WiC_biSentence, WiC_biSentence : 모델 2개에 seq 하나씩 각각 들어가는 Siamese-BERT 구조.

countEpoch = 0

#bestMCC = -2 #-1 ~ +1
bestAcc = -1 #0 ~ 1
bestLoss = 1 #0 ~ 1
bestLoss_at = 0

class CosineLoss(nn.Module):
    def __init__(self, xent=.1, reduction="mean"):
        super(CosineLoss, self).__init__()
        self.xent = xent
        self.reduction = reduction
        
        self.y = torch.Tensor([1])
        
    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=self.reduction)
        cent_loss = F.cross_entropy(F.normalize(input), target, reduction=self.reduction)
        
        return cosine_loss + self.xent * cent_loss

def train_wic_uniModel(model, data_loader, batch_size, epochs, lf, optimizer, device):
    model.train() #set model training mode
    min_loss = 1
    for _ in range(epochs):
        correct = 0
        all_loss = []
        mini_batch = 0
        print(f'[epoch {countEpoch+_}]')
        
        #uni-Bert: WiC_uniSentence(dataset), model_WiC_uniSent(model)
        for batchIdx, (input_ids, token_type_ids, attention_mask, label, tokensIdx1, tokensIdx2) in enumerate(data_loader):
            #print('[epoch, batch_index]',_ ,batchIdx) #

            #device = torch.device('cuda:0')
            #model.to(device)
            model.zero_grad() #model weight 초기화 맞나?

            #Sent_token
            input_ids = input_ids.to(device)#.cuda() #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)#.cuda() #
            attention_mask = attention_mask.to(device)#.cuda() #

            label = label.long().to(device)#.cuda()
            tokensIdx1 = tokensIdx1.long().to(device)#.cuda()
            tokensIdx2 = tokensIdx2.long().to(device)#.cuda()

            #print('model input(iids):',input_ids.shape) #torch.Size([10, 20])
            output = model(input_ids, token_type_ids, attention_mask, tokensIdx1, tokensIdx2) #shape: out(torch.Size([bs, 2])), label
            #print('model output(all):',output.shape) #torch.Size([10, 2])
            #print('output shape :',output.shape) #torch.Size([bs, 2])
            #print('label shape: ',label.shape) #torch.Size([50])

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
        accuracy = (correct / len(wicDataset_train)).float()
        print("acc = ", accuracy,", loss = ",avg_loss)

        min_loss = min(min_loss, avg_loss)

    return min_loss

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
        for batchIdx, (input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2, label, tokenIdx1_s, tokenIdx1_e, tokenIdx2_s, tokenIdx2_e) in enumerate(data_loader):
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
            tokenIdx1_s = tokenIdx1_s.long().to(device)
            tokenIdx1_e = tokenIdx1_e.long().to(device)
            tokenIdx2_s = tokenIdx2_s.long().to(device)
            tokenIdx2_e = tokenIdx2_e.long().to(device)

            '''
            #print('model input(iids):',input_ids.shape) #torch.Size([10, 20])
            output = model(input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2, tokensIdx1, tokensIdx2) #shape: out(torch.Size([bs, 2])), label

            lf_input=output#.view_as(label) #output
            lf_target=label#[batchIdx] #label
            print('loss_input shape: ',lf_input.shape) #torch.Size([10, 2])
            print('loss_target shape: ',lf_target.shape) #torch.Size([10])

            loss = lf(output,label) #lf(output,label)
            pred = torch.argmax(output,dim=-1)
            '''

            #output1 = model(input_ids1, token_type_ids1, attention_mask1, tokenIdx1_s, tokenIdx1_e) #shape: [bs,768]
            #output2 = model(input_ids2, token_type_ids2, attention_mask2, tokenIdx2_s, tokenIdx2_e) #shape: [bs,768]
            output = model(input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2, tokenIdx1_s, tokenIdx1_e, tokenIdx2_s, tokenIdx2_e) #shape: [bs,768]
            #print('train model 1,2 shape: ', output1.shape, output2.shape)
            #output = torch.cat([output1, output2], dim=1) #(bs,1)*2 -> (bs*2)
            #print('train output shape:', output.shape, output[0], label[0]) #output(tensor[5.xx,...,-1.xx]*768), label(tensor[0/1])
            #print(f'shape - output:{output.shape}, label:{label.shape}') #output(tensor[bs,768*2]), label(tensor[bs])

            loss = lf(output,label) #lf(pred,label) #lf(output,label)
            #customLoss = 
            pred = torch.argmax(output,dim=-1)
            

            #print(f'types: pred {pred.shape}{type(pred)}, label {label.shape}{type(label)}, loss {type(loss)}') 
            ##pred torch.Size([50])<class 'torch.Tensor'>, label torch.Size([50])<class 'torch.Tensor'>, loss <class 'torch.Tensor'>

            #input = torch.randn(3, 5, requires_grad=True)
            #target = torch.empty(3, dtype=torch.long).random_(5)
            #print('example shape: ', input.shape, target.shape, lf(input, target).shape) #torch.Size([3, 5]) torch.Size([3]) -> torch.Size([])

            #print(f'output={output.shape}, label={label.detach().cpu().shape}') # torch.Size([50, 1]), torch.Size([50])
            #print(f'loss={loss}, pred={pred.detach().cpu().shape}') # 
            correct += sum(pred.detach().cpu()==label.detach().cpu())
            all_loss.append(loss)
            loss.backward() #기울기 계산
            optimizer.step() #가중치 업데이트
            mini_batch += batch_size

            if mini_batch%1000 == 0:
                print(f'batch{mini_batch}:',end='')
                accuracy = eval_wic_biModel(mymodel, eval_loader, bs, device)
                acc_list.append(accuracy)
                model.train() #set model training mode
            #print(mini_batch,"/",len(colaDataset)) #전체 Dataset 중 batch 단위로 수행완료. 

        #print(sum(all_loss)/len(all_loss))
        #print("acc = ", correct / len(colaDataset))
        avg_loss = (sum(all_loss)/len(all_loss)).detach().cpu().float()
        accuracy = (correct / len(wicDataset_train)).float()
        print("acc = ", accuracy,", loss = ",avg_loss)

        min_loss = min(min_loss, avg_loss)

    return min_loss


def eval_wic_uniModel(model, data_loader, batch_size, device):
    model.eval()

    y_true = None #label list
    y_pred = None #model prediction list

    #print('len dataloader:',len(data_loader))
    #testM = torch.randn(2,3) #torch.Size([2, 3])
    #print(testM, testM.shape)
    #print(torch.argmax(testM,dim=0).shape) #col레벨에서 큰 숫자의 index, torch.Size([3])
    #print(torch.argmax(testM,dim=1)) #row 중에서 큰숫자 index, torch.Size([2])
    #print(torch.argmax(testM,dim=-1)) #row 중에서 큰숫자 index, torch.Size([2]). 완벽히 같음.
    #sm1 = nn.Softmax(dim=0) #세로합=1
    #print(sm1(testM))
    #sm2 = nn.Softmax(dim=1) #가로합=1
    #print(sm2(testM))

    for batchIdx, (input_ids, token_type_ids, attention_mask, label, tokensIdx1, tokensIdx2) in enumerate(data_loader):
        with torch.no_grad(): #autograd 끔->속도향상. 사실 model.eval()하면 안해줘도 됨.
            input_ids = input_ids.to(device)#.cuda() #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.long().to(device)
            tokensIdx1 = tokensIdx1.long().to(device)
            tokensIdx2 = tokensIdx2.long().to(device)

            output = model(input_ids, token_type_ids, attention_mask, tokensIdx1, tokensIdx2)
            #print('loss_input shape: ',output.shape) #torch.Size([10, 2]) #batch마다 한번씩.
            #print(batchIdx,'output shape:',output.shape) #torch.Size([bs, 2])
            #print(torch.argmax(output,dim=1).shape) #dim=0(2), dim=-1(bs)
            #logits = output#torch.argmax(output,dim=1) #output

            #eval_loss += tmp_eval_loss.mean().item()
        #nb_eval_steps += 1

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

def eval_wic_biModel(model, data_loader, batch_size, device):
    model.eval()

    y_true = None #label list
    y_pred = None #model prediction list

    for batchIdx, (input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2, label, tokenIdx1_s, tokenIdx1_e, tokenIdx2_s, tokenIdx2_e) in enumerate(data_loader):
        with torch.no_grad(): #autograd 끔->속도향상. 사실 model.eval()하면 안해줘도 됨.
            input_ids1 = input_ids1.to(device) #move param_buffers to gpu
            token_type_ids1 = token_type_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            input_ids2 = input_ids2.to(device)
            token_type_ids2 = token_type_ids2.to(device)
            attention_mask2 = attention_mask2.to(device)
            
            label = label.long().to(device)
            tokenIdx1_s = tokenIdx1_s.long().to(device)
            tokenIdx1_e = tokenIdx1_e.long().to(device)
            tokenIdx2_s = tokenIdx2_s.long().to(device)
            tokenIdx2_e = tokenIdx2_e.long().to(device)

            #print(input_ids1.shape, token_type_ids1.shape, attention_mask1.shape) #torch.Size([bs, 100]) * 3
            #print(input_ids2.shape, token_type_ids2.shape, attention_mask2.shape) #torch.Size([bs, 100]) * 3

            output1 = model(input_ids1, token_type_ids1, attention_mask1, tokenIdx1_s, tokenIdx1_e)
            output2 = model(input_ids2, token_type_ids2, attention_mask2, tokenIdx2_s, tokenIdx2_e)
            #print('output1,2 shape:', output1.shape, output2.shape)
            output = torch.cat([output1, output2], dim=1) #(bs,1)*2 -> (bs*2)
            #print('output shape:', output.shape, output[0], label[0])
            
            #output = model(input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2, tokensIdx1, tokensIdx2)
            #print('loss_input shape: ',output.shape) #torch.Size([10, 2]) #batch마다 한번씩.
            #print('model output: ',output, output.shape,' -> ', torch.argmax(output,dim=1).shape)
            #logits = torch.argmax(output,dim=1) #output #argmax가 안되고 있는거같아서 한번 체크하기.

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


def inference_wic_uniModel(model, data_loader, batch_size, device):
    model.eval()

    y_pred = None #model prediction list
    for batchIdx, (input_ids, token_type_ids, attention_mask, label, tokensIdx1, tokensIdx2) in enumerate(data_loader):
        with torch.no_grad(): #autograd 끔->속도향상. 사실 model.eval()하면 안해줘도 됨.
            input_ids = input_ids.to(device)#.cuda() #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.long().to(device)
            tokensIdx1 = tokensIdx1.long().to(device)
            tokensIdx2 = tokensIdx2.long().to(device)

            output = model(input_ids, token_type_ids, attention_mask, tokensIdx1, tokensIdx2)

        if y_pred is None:
            y_pred = output.detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, output.detach().cpu().numpy(), axis=0)
    y_pred = np.argmax(y_pred, axis=1)

    print('output shape: ',y_pred.shape, type(y_pred))

    return y_pred

def inference_wic_biModel(model, data_loader, batch_size, device):
    model.eval()

    y_pred = None #model prediction list

    for batchIdx, (input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2, label, tokenIdx1_s, tokenIdx1_e, tokenIdx2_s, tokenIdx2_e) in enumerate(data_loader):
        with torch.no_grad(): #autograd 끔->속도향상. 사실 model.eval()하면 안해줘도 됨.
            input_ids1 = input_ids1.to(device) #move param_buffers to gpu
            token_type_ids1 = token_type_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            input_ids2 = input_ids2.to(device)
            token_type_ids2 = token_type_ids2.to(device)
            attention_mask2 = attention_mask2.to(device)
            
            label = label.long().to(device)
            tokenIdx1_s = tokenIdx1_s.long().to(device)
            tokenIdx1_e = tokenIdx1_e.long().to(device)
            tokenIdx2_s = tokenIdx2_s.long().to(device)
            tokenIdx2_e = tokenIdx2_e.long().to(device)

            output1 = model(input_ids1, token_type_ids1, attention_mask1, tokenIdx1_s, tokenIdx1_e)
            output2 = model(input_ids2, token_type_ids2, attention_mask2, tokenIdx2_s, tokenIdx2_e)
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
    #model_savePth = homePth+'/model_pth/kobert_WiC/'
    #model_loadPth = homePth+'/model_pth/kobert_WiC/'
    print('homePth:',homePth,', curPth:',os.getcwd())
    start_day_time=datetime.now().strftime("%m/%d,%H:%M:%S")
    print('training start at (date,time): ',start_day_time)
    
    #dataPth_dev = datasetPth+'task2_homonym/NIKL_SKT_WiC_Train.tsv'
    tsvPth_train = datasetPth+taskDir_path+fname_train #'task2_homonym/NIKL_SKT_WiC_Train.tsv'
    tsvPth_dev = datasetPth+taskDir_path+fname_dev #'task2_homonym/NIKL_SKT_WiC_Dev.tsv'
    tsvPth_test = datasetPth+taskDir_path+_#fname_test #'task2_homonym/NIKL_SKT_WiC_Test_labeled.tsv'

    bs = 50#50 #100 #10,20,100,200
    epochs= 0#100 #10
    num_printEval = 1#2 #꼭 epochs의 약수가 되게 넣어주기. 안그럼 epochs가 조금 모자라게 돔.

    device = torch.device('cuda:0')
    model_type = 'biBERT' # 'uniBert', 'biBERT'
    random_seed_int = 5 # 랜덤시드 넘버=5 로 고정
    set_seed(random_seed_int, device) #random seed 정수로 고정.

    bool_save_model, bool_load_model, bool_save_output = False, False, False #default: True, False, False
    model_savePth = homePth+f'/model_pth/{model_name}_{task_name}/'
    model_loadPth = homePth+f'/model_pth/{model_name}_{task_name}/'
    filename_modelSave_chkpt = f'modelpth_{task_name}_{model_type}_epoch{countEpoch}by{num_printEval}_bs{bs}.pt'
    filename_modelLoad_chkpt = f'd1022_t_modelpth_{task_name}_{model_type}_epoch50_bs100.pt' #f'modelpth_{task_name}_{model_type}_epoch{epochs}by{num_printEval}_bs{bs}.pt'
    #modelOutputPth = filename_modelLoad_chkpt+f'_xTrain_epc{epochs}by{num_printEval}_bs{bs}.json' #json file
    print('save_model:',bool_save_model,'filename_modelSave_chkpt: ',filename_modelSave_chkpt)
    print('load_model:',bool_load_model,'filename_modelLoad_chkpt: ',filename_modelLoad_chkpt)

    lf = nn.CrossEntropyLoss()
    #lf = nn.CosineEmbeddingLoss()
    #lf = nn.MSELoss()
    #lf = CosineLoss()

    mymodel = model_WiC_biSent() #default: bi-BERT
    if not bool_load_model and model_type == 'uniBERT':
        mymodel = model_WiC_uniSent() #model_WiC_uniSent(uni-Bert) 
    elif not bool_load_model and model_type == 'biBERT':
        pass
        #mymodel = model_WiC_biSent() #model_WiC_biSent(Siamese-BERT)
    elif bool_load_model:
        mymodel = load_model(model_loadPth, filename_modelLoad_chkpt, mymodel, device)
    print('mymodel is biBERT: ',hasattr(mymodel,'cosSim'),', load_model: ',bool_load_model) #cosSim유무로 bi/uni model여부 판단. 
    mymodel.to(device)

    optimizer = Adam(mymodel.parameters(),lr=2e-5, eps=1e-8)

    #uni-Bert: WiC_uniSentence(dataset), model_WiC_uniSent(model)
    #Siamese-BERT: WiC_biSentence(dataset), model_WiC_biSent(model)
    wicDataset_train = WiC_biSentence(os.path.join(os.getcwd(),tsvPth_train)) #dataPth_train(default: bi-BERT)
    wicDataset_dev = WiC_biSentence(os.path.join(os.getcwd(),tsvPth_dev)) #dataPth_dev(default: bi-BERT)
    wicDataset_test = WiC_biSentence(os.path.join(os.getcwd(),tsvPth_test)) #dataPth_test(default: bi-BERT)
    if model_type == 'uniBERT':
        wicDataset_train = WiC_uniSentence(os.path.join(os.getcwd(),tsvPth_train)) #(dataPth_train)
        wicDataset_dev = WiC_uniSentence(os.path.join(os.getcwd(),tsvPth_dev)) #dataPth_dev(default: bi-BERT)
        wicDataset_test = WiC_uniSentence(os.path.join(os.getcwd(),tsvPth_test)) #(dataPth_dev)
    elif model_type == 'biBERT':
        pass
    TrainLoader = DataLoader(wicDataset_train, batch_size=bs)
    EvalLoader = DataLoader(wicDataset_dev, batch_size=bs)
    InferenceLoader = DataLoader(wicDataset_test, batch_size=bs)
    print('wicDataset_train is biBERT dataset: ',wicDataset_train.__name__=='WiC_biSentence',', fileName:',tsvPth_train) #__name__ 값으로 bi/uni model 여부 판단
    print('wicDataset_dev is biBERT dataset: ',wicDataset_dev.__name__=='WiC_biSentence',', fileName:',tsvPth_dev) #__name__ 값으로 bi/uni model 여부 판단

    print('[Training Phase]')
    print(f'len WiC_train:{len(wicDataset_train)}, batch_size:{bs}, epochs:{epochs}(eval_by {num_printEval}), model:{model_type}, device({device})')
    for epoch in range(int(epochs/num_printEval)):
        accuracy = bestAcc #eval_wic(mymodel, EvalLoader, bs, device)
        #bestAcc = max(bestAcc,accuracy)
        minLoss = bestLoss
        if model_type == 'uniBERT':
            accuracy = eval_wic_uniModel(mymodel, EvalLoader, bs, device)
            minLoss = train_wic_uniModel(mymodel, TrainLoader, bs, num_printEval, lf, optimizer, device) #3epoch마다 eval
        elif model_type == 'biBERT':
            accuracy = eval_wic_biModel(mymodel, EvalLoader, bs, device)
            minLoss = train_wic_biModel(mymodel, TrainLoader, bs, num_printEval, lf, optimizer, device) #3epoch마다 eval
        bestAcc = max(bestAcc,accuracy)
        #bestLoss = min(bestLoss,minLoss)
        if minLoss < bestLoss:
            bestLoss = minLoss
            bestLoss_at = countEpoch
        countEpoch+=num_printEval #countEpoch 업데이트

        if bool_save_model: #epoch마다 모델 weight 저장.(총 epochs.현재epoch)
            filename_modelSave_chkpt = f'{task_name}_{model_type}_epoch{epochs}.{countEpoch}by{num_printEval}_bs{bs}_rs{random_seed_int}.pt'
            save_model(model_savePth, filename_modelSave_chkpt, mymodel, optimizer)
    #train_wic(mymodel, TrainLoader, bs, epochs, lf, optimizer, device)

    print('[Evaluation Phase]')
    print(f'len WiC_dev:{len(wicDataset_dev)}, batch_size:{bs}, epochs:{epochs}, model:{model_type}, device({device})')
    result = 0
    if model_type == 'uniBERT':
        result = eval_wic_uniModel(mymodel, EvalLoader, bs, device)
    elif model_type == 'biBERT':
        result = eval_wic_biModel(mymodel, EvalLoader, bs, device)
    #result = eval_wic(mymodel, EvalLoader, bs, device)
    bestAcc = max(bestAcc,result)

    print('[Inference Phase]')
    print(f'len {task_name}_test:{len(wicDataset_dev)}, batch_size:{bs}, epochs:{epochs}, model:{model_type}, device({device})')
    #InferenceLoader = DataLoader(wicDataset_test, batch_size=bs)
    if model_type == 'uniBERT':
        modelOutput = inference_wic_uniModel(mymodel, InferenceLoader, bs, device)
    elif model_type == 'biBERT':
        eval_wic_biModel(mymodel, InferenceLoader, bs, device)
        modelOutput = inference_wic_biModel(mymodel, InferenceLoader, bs, device)

    end_day_time=datetime.now().strftime("%m/%d, %H:%M:%S") #Date %m/%d %H:%M:%S
    print(f'training model from {start_day_time} to {end_day_time} (date,time): ')

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

    '''
    ## mymodel, optimizer 오브젝트: {key:value}, {key:{valueDict.key:valueDict.value}}
    print('model: ',len(mymodel.state_dict()), type(mymodel), ', optimizer: ',len(optimizer.state_dict()), type(optimizer)) #<class 'kobert_models.model_WiC_uniSent'>
    print('model state_dict len: ',len(mymodel.state_dict())) #208
    for idx, (key, value) in enumerate(zip(mymodel.state_dict().keys(),mymodel.state_dict().values())):
        #print('model dict(key,value) type: ',type(key), type(value_dict)) # <class 'str'> <class 'torch.Tensor'>
        print(f'{idx}th ({key}, {value.shape})') #
    print('optimizer state_dict len: ',len(optimizer.state_dict())) #2
    print(type(optimizer.state_dict().keys()), optimizer.state_dict().keys(), len(optimizer.state_dict().keys())) #<class 'dict_keys'> dict_keys(['state', 'param_groups'])
    for key, value_dict in zip(optimizer.state_dict().keys(),optimizer.state_dict().values()):
        #print('optimizer(key,val_dict): ',key, value_dict.keys()) #value_dict={0~206}
        for val_i, (value_key, value_val) in enumerate(zip(value_dict.keys(),value_dict.values())):
            #print(f'{idx}.{val_i}th key:{value_key}, value_keys{value_val.keys()}') #value_keysdict_keys(['step', 'exp_avg', 'exp_avg_sq'])
            #print([type(vals) for vals in value_val.values()]) #[<class 'int'>, <class 'torch.Tensor'>, <class 'torch.Tensor'>]
            mykey1, mykey2 ='exp_avg', 'exp_avg_sq'
            print(f'{idx}.{val_i}th key(exp_avg):{value_val[mykey1].shape}, key(exp_avg_sq):{value_val[mykey2].shape}')
    '''


    print('finish')
    print('<SUMMARY>')
    print(f'task:{task_name}, model:{model_name}({model_type}), bs:{bs}, epochs:{epochs}, load/save model:{bool_load_model}/{bool_save_model}, randSeedNum:{random_seed_int}')
    if bool_load_model:
        print(f'model chkpoint_loadPth:{model_loadPth}')
    if bool_save_model:
        print(f'model chkpoint_savePth:{model_savePth}')
    print(f'bestAccuracy:{bestAcc}, bestLoss:{bestLoss}(bestLoss around epoch {bestLoss_at})')




