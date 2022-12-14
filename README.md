# On-Device Audio-Visual  Multi-Person Wake Word Spotting

CAAI Transactions on Intelligence Technology

# Requirements

- dlib==19.7.0
face-alignment==1.3.5
face-recognition==1.3.0
face-recognition-models==0.3.0
ffmpeg==1.4
Flask==2.0.1
Flask-Login==0.5.0
librosa==0.7.2
moviepy==1.0.3
numpy==1.19.5
opencv-python==4.5.4.60
Pillow==8.1.2
PyAudio==0.2.11
Python==3.6.12
scikit-image==0.17.2
scikit-learn==0.22.2
scikit-video==1.1.11
scipy==1.5.4
SoundFile==0.10.3.post1
tensorboard==2.8.0
torch==1.6.0+cu101
torchaudio==0.10.2
torchvision==0.9.1
tqdm==4.63.1
transformers==4.14.1
wandb==0.12.18
Wave==0.0.2
wget==3.2


# Preprocess
 First, you need to download PKU-KWS dataset from https://zenodo.org/record/6792058. 
 Second, put the dataset into D://datasets

The PKU-KWS dataset is collected in a relatively quiet acoustic environment with a camera recording at the speed of 25 frames per second. The video resolution is 1080 × 1920, and the audio is synchronously recorded at the sampling frequency of 16000Hz, with 16 bits for each sampling. 

 After you prepare the dataset, run `python preprocess_audio.py` and `python preprocess_video.py` to get the preprocess data.
 You can download processed PKU-KWS dataset from https://disk.pku.edu.cn:443/link/5CC8F2333B5A430EE12C8536CC2EFBD9
 
# Training for VAD Model
You can run `python train.py` in the **vad** folder to train the VAD model, and then please run `python preprocess_video.py` in the vad folder to do Person-face selection.

# Training for Teacher Model(VAD+AV-WWS)

You can run `python train.py` in the **w+vo** folder to train the VO-WWS model;
You can run `python train.py` in the **w+ao** folder to train the AO-WWS model;
You can run `python train.py` in the **w+av** folder to train the AV-WWS model.

> **WARNING**: You have to train on single GPU.

# Test for Teacher Model(VAD+AV-WWS)

You can run `python test.py` in the **w+vo** folder to test the VO-WWS model;
You can run `python test.py` in the **w+ao** folder to test the AO-WWS model;
You can run `python test.py` in the **w+av** folder to test the AV-WWS model.


# Training for Student Model(VAD+AV-WWS+KD)

You can run `python train.py` in the **kd** folder to test the student model;

# Testing for Student Model(VAD+AV-WWS+KD)

You can run `python test.py` in the **kd** folder to test the student model;

> We provide teacher models and student models for testing code. You can download them for free.

teacher-model：
https://disk.pku.edu.cn:443/link/4527F96A7F448C70CB00D94989E0E0F0

student-model：
https://disk.pku.edu.cn:443/link/3E8C441C19B48CF52DA80CABCDB5CD3E

> We also provide the fairseq library for calling wav2dev pretrained models at:
https://disk.pku.edu.cn:443/link/89E6973AE8EDD30C15931B468C4F78F4

# Visualization

You can run `python test_flask.py` in the **choose** folder to visualize the whole process of our model.

> We also provide baseline and reproduced MCNN code. You can use it freely by running `python train.py` in the baseline and mcnn folders.
"# mutil-person_avwws" 
