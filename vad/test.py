
import numpy as np
from general import *
from mymodel.SWWNetModel import *
import torch
from preprocess_audio import preprocess_audio
from detect_landmarks_in_image import concat_pic
from preprocess_video import preprocess_video

def test(file):
    len = concat_pic(file)
    preprocess_video(len)



    np.random.seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gpuAvailable = torch.cuda.is_available()
    device = torch.device('cuda:0' if gpuAvailable else 'cpu')



    model = SWWNetModel()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(r'C:\Users\WGQ\Desktop\qtchoose\static/1.pt')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)


    audioFeature,visualFeature = preprocess_audio(file,len)


    audioFeature = audioFeature.to(device)
    visualFeature = visualFeature.to(device)
    model.eval()

    with torch.no_grad():
        model.eval()
        out_wake_word = model(audioFeature, visualFeature)
        out_wake_word = F.softmax(out_wake_word, dim=1)
        _, preds = torch.max(out_wake_word, 1)
        preds = preds.cpu().numpy()
        out_wake_word = out_wake_word.cpu().numpy()
        print('preds',preds)
        print('out word',out_wake_word)
        return preds,out_wake_word

#test(r'D:\199\C0456W000100E000R001S01O01.mp4')