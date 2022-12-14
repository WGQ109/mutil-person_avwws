
import torch.nn as nn
from dataLoader import *
from mymodel.SWWNetModel import SWWNetModel
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

criterion = nn.CrossEntropyLoss().cuda()

def loss_function(x_input,y_target):
    log_output = torch.log(x_input)

    nllloss_func = nn.NLLLoss().cuda()
    nlloss_output = nllloss_func(log_output, y_target)

    return nlloss_output


def train_network(model, loader, optim, device):
    running_corrects = 0.0
    running_all = 0.0
    loss = 0.0
    for num, (audioFeature, visualFeature, wake_word_labels, mask,a) in enumerate(
            tqdm(loader, leave=False, desc="Train", ncols=75)):
        audioFeature = audioFeature.to(device)
        visualFeature = visualFeature.to(device)
        mask = mask.to(device)
        mask = mask.byte()
        mask = mask.bool()
        b = (mask == False)

        wake_word_labels = wake_word_labels.to(device)
        wake_word_labels = wake_word_labels.reshape(-1)

        optim.zero_grad()
        model.train()

        stu_out_wake_word, stu_o = model(audioFeature, visualFeature)
        stu_out_wake_word = F.softmax(stu_out_wake_word, dim=1)
        stu_o = F.softmax(stu_o, dim=1)
        out_wake_word = stu_out_wake_word * 0.7 + stu_o * 0.7

        nloss = loss_function(out_wake_word, wake_word_labels)

        _, preds = torch.max(out_wake_word.data, 1)

        running_corrects += torch.sum(preds == wake_word_labels.data)
        running_all += len(out_wake_word)
        nloss.backward()
        optim.step()
        loss += nloss.item()

    print('test acc:%.4f' % (running_corrects / running_all))

    return loss / num, running_corrects / running_all


def evaluate_network(model, loader, device):
    loss = 0.0
    running_corrects = 0.0
    running_all = 0.0
    list = []
    list_out = []

    for num, (audioFeature, visualFeature, wake_word_labels,mask, a) in enumerate(
            tqdm(loader, leave=False, desc="Eval", ncols=75)):

        audioFeature = audioFeature.to(device)
        visualFeature = visualFeature.to(device)
        wake_word_labels = wake_word_labels.to(device)
        wake_word_labels = wake_word_labels.reshape(-1)
        mask = mask.to(device)
        mask = mask.byte()
        mask = mask.bool()
        b = (mask == False)
        model.eval()

        for number in a:
            elem = number[0][6:11]
            list.append([int(label) for label in elem])

        with torch.no_grad():
            model.eval()
            out_wake_word,o = model(audioFeature, visualFeature)
            out_wake_word = out_wake_word*0.7+o*0.7
            _, preds = torch.max(F.softmax(out_wake_word, dim=1), 1)
            nloss = criterion(out_wake_word, wake_word_labels)
            running_corrects += torch.sum(preds == wake_word_labels.data)

            b = F.softmax(out_wake_word, dim=1)
            b = b.cpu().tolist()
            for out in b:
                list_out.append(out)
            running_all += len(out_wake_word)
            loss += nloss.item()


    return loss / num, running_corrects / running_all

