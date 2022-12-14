from general import *
import matplotlib.pyplot as plt
from mymodel.SWWNetModel import *
import torch
import shutil
import os
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main():
    np.random.seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    gpuAvailable = torch.cuda.is_available()
    device = torch.device('cuda:0' if gpuAvailable else 'cpu')

    if os.path.exists(r'D:/get/talk' + '/checkpoints'):
        shutil.rmtree(r'D:/get/talk' + '/checkpoints')

    os.mkdir(r'D:/get/talk' + '/checkpoints')
    os.mkdir(r'D:/get/talk' + '/checkpoints/models')
    os.mkdir(r'D:/get/talk' + '/checkpoints/plots')

    audio_path = r'D:/dataset/new_audio_wav'

    video_path = r'D:/dataset/new_video_npy1'
    dsets = {x: MyDataSet(x, audio_path, video_path) for x in ['train', 'val']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=8, shuffle=True, num_workers=0,
                                                   collate_fn=mycollate_fn) for x in ['train', 'val']}
    model = SWWNetModel()
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size=1,gamma=0.95)


    print(optim.state_dict()['param_groups'][0]['lr'])



    trainingLossCurve = list()
    validationLossCurve = list()
    epoch = 1
    logfilename = '1' + '.txt'
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfilename, mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    while(1):
        trainingLoss,trainingAcc = train_network(model,dset_loaders['train'],optim,device)
        trainingLossCurve.append(trainingLoss)
        validationLoss,validationAcc= evaluate_network(model,dset_loaders['val'],device)
        validationLossCurve.append(validationLoss)

        logger.info("Step:%03d ||Tr.Loss:%.6f Acc.Loss: %.6f" % (epoch, trainingLoss, validationAcc))

        if epoch >= 1000:
            quit()
        #print("第%d个epoch的学习率：%f" % (epoch, optim.param_groups[0]['lr']))
        scheduler.step()
        print(optim.state_dict()['param_groups'][0]['lr'])

        if ((epoch % 2 == 0) or (epoch == 1000)) and (epoch != 0):

            savePath = r'D:/get/talk' + '/checkpoints/models/train-step_{:04d}-loss_{:.3f}.pt'.format(epoch,validationAcc)
            torch.save(model.state_dict(),savePath)

            plt.figure()
            plt.title('Loss Curves')
            plt.xlabel('Step No.')
            plt.ylabel('Loss value')
            plt.plot(list(range(1, len(trainingLossCurve) + 1)),trainingLossCurve,'blue',label='train')
            plt.plot(list(range(1, len(validationLossCurve) + 1)), validationLossCurve, 'red', label='Validation')
            plt.legend()
            plt.savefig(r'D:/get/talk' + '/checkpoints/plots/train-step_{:04d}-loss.png'.format(epoch))
            plt.close()

        epoch += 1


if __name__ == '__main__':
    main()
