
import torch.nn as nn
from preprocess_audio import *
from mymodel.SWWNetModel import SWWNetModel
import torch.nn.functional as F
from tqdm import tqdm

criterion = nn.CrossEntropyLoss().cuda()

def train_network(model, loader, optim ,device):

    running_corrects = 0.0
    running_all = 0.0
    loss = 0.0
    for num,(audioFeature, visualFeature, wake_word_labels,a) in enumerate(tqdm(loader,leave=False,desc="Train", ncols=75)):
        audioFeature = audioFeature.to(device)
        visualFeature = visualFeature.to(device)
        wake_word_labels = wake_word_labels.to(device)
        wake_word_labels = wake_word_labels.reshape(-1)


        optim.zero_grad()
        model.train()
        out_wake_word = model(audioFeature,visualFeature)

        _, preds = torch.max(out_wake_word.data, 1)
        nloss = criterion(out_wake_word, wake_word_labels)

        '''
        L1_reg = 0
        for param in model.parameters():
            L1_reg += torch.sum(torch.abs(param))
        nloss += 0.000001 * L1_reg  # lambda=0.001
        '''

        running_corrects += torch.sum(preds == wake_word_labels.data)
        running_all += len(out_wake_word)
        nloss.backward()
        optim.step()
        loss += nloss.item()

    print('test acc:%.4f' % (running_corrects/running_all))

    return loss/num,running_corrects/running_all

def evaluate_network(model,loader,device):

    loss = 0.0
    running_corrects = 0.0
    running_all = 0.0
    for num, (audioFeature, visualFeature, wake_word_labels,a) in enumerate(tqdm(loader, leave=False, desc="Eval", ncols=75)):

        audioFeature = audioFeature.to(device)
        visualFeature = visualFeature.to(device)
        wake_word_labels = wake_word_labels.to(device)
        wake_word_labels = wake_word_labels.reshape(-1)
        model.eval()
        with torch.no_grad():
            model.eval()
            out_wake_word = model(audioFeature, visualFeature)
            _, preds = torch.max(F.softmax(out_wake_word, dim=1), 1)
            nloss = criterion(out_wake_word, wake_word_labels)
            running_corrects += torch.sum(preds == wake_word_labels.data)
            '''
            L1_reg = 0
            for param in model.parameters():
                L1_reg += torch.sum(torch.abs(param))
            nloss += 0.000001 * L1_reg  # lambda=0.001
            '''
            '''
            
            if not all(preds.cpu().numpy() == wake_word_labels.cpu().numpy()):
                print()
                print(preds,wake_word_labels,a)
            '''
            running_all += len(out_wake_word)
            loss += nloss.item()

    print('eval acc:%.4f' % (running_corrects / running_all))
    return loss / num,running_corrects/running_all