
'''
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
            out_wake_word,o = model(audioFeature, visualFeature,b)
            out_wake_word = out_wake_word*0.5+o*0.5
            _, preds = torch.max(F.softmax(out_wake_word, dim=1), 1)
            nloss = criterion(out_wake_word, wake_word_labels)
            running_corrects += torch.sum(preds == wake_word_labels.data)

            b = F.softmax(out_wake_word, dim=1)
            b = b.cpu().tolist()
            for out in b:
                list_out.append(out)
            running_all += len(out_wake_word)
            loss += nloss.item()

    list = np.array(list)
    list_out = np.array(list_out)
    print('eval acc:%.4f' % (running_corrects / running_all))
    y_label = list
    y_score = list_out
    n_classes = 5

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    fnr = dict()
    eer = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
        fnr[i] = 1 - tpr[i]
        fnr[i] = fnr[i] * 100
        fpr[i] = fpr[i] * 100
        idxE = np.nanargmin(np.absolute((fnr[i] - fpr[i])))
        if fpr[i][idxE] > fnr[i][idxE]:
            eer[i] = fpr[i][idxE]
        else:
            eer[i] = fnr[i][idxE]
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    fnr["micro"] = 1 - tpr["micro"]
    fnr["micro"] = fnr["micro"] * 100
    fpr["micro"] = fpr["micro"] * 100
    idxE = np.nanargmin(np.absolute((fnr["micro"] - fpr["micro"])))
    if fpr["micro"][idxE] > fnr["micro"][idxE]:
        eer["micro"] = fpr["micro"][idxE]
    else:
        eer["micro"] = fnr["micro"][idxE]
    print(eer)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # macro（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print(roc_auc)
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-calss ROC')
    plt.legend(loc="lower right")
    plt.show()

    return loss / num, running_corrects / running_all

'''


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
from scipy import interp
criterion = nn.CrossEntropyLoss().cuda()




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
        out_wake_word,o = model(audioFeature, visualFeature,b)
        out_wake_word = o
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
            out_wake_word,o = model(audioFeature, visualFeature,b)
            out_wake_word = o
            _, preds = torch.max(F.softmax(out_wake_word, dim=1), 1)
            nloss = criterion(out_wake_word, wake_word_labels)
            running_corrects += torch.sum(preds == wake_word_labels.data)

            b = F.softmax(out_wake_word, dim=1)
            b = b.cpu().tolist()
            for out in b:
                list_out.append(out)
            running_all += len(out_wake_word)
            loss += nloss.item()

    list = np.array(list)
    list_out = np.array(list_out)
    print('eval acc:%.4f' % (running_corrects / running_all))
    y_label = list
    y_score = list_out
    n_classes = 5

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    fnr = dict()
    eer = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
        fnr[i] = 1 - tpr[i]
        fnr[i] = fnr[i] * 100
        fpr[i] = fpr[i] * 100
        idxE = np.nanargmin(np.absolute((fnr[i] - fpr[i])))
        if fpr[i][idxE] > fnr[i][idxE]:
            eer[i] = fpr[i][idxE]
        else:
            eer[i] = fnr[i][idxE]
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    fnr["micro"] = 1 - tpr["micro"]
    fnr["micro"] = fnr["micro"] * 100
    fpr["micro"] = fpr["micro"] * 100
    idxE = np.nanargmin(np.absolute((fnr["micro"] - fpr["micro"])))
    if fpr["micro"][idxE] > fnr["micro"][idxE]:
        eer["micro"] = fpr["micro"][idxE]
    else:
        eer["micro"] = fnr["micro"][idxE]
    print(eer)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # macro（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print(roc_auc)
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-calss ROC')
    plt.legend(loc="lower right")
    plt.show()

    return loss / num, running_corrects / running_all


if __name__ == '__main__':
    # -*-coding:utf-8-*-

    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc
    from scipy import interp

    y_label = np.array([
        [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [0, 1, 0], [0, 1, 0], [0, 1, 0],
        [0, 0, 1], [0, 0, 1], [0, 0, 1]
    ])

    y_score = np.array([
        [0.8, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.1, 0.3],
        [0.2, 0.5, 0.3], [0.1, 0.6, 0.3], [0.2, 0.75, 0.05],
        [0.05, 0.05, 0.9], [0.1, 0.3, 0.6], [0.12, 0.8, 0.08],
    ])

    n_classes = 3

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print(roc_auc)
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-calss ROC')
    plt.legend(loc="lower right")
    plt.show()