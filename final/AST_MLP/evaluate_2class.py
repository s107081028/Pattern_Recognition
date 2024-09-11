import argparse
import pickle
import sys
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import os
import dataloader_test as dataloader
from ast_models_tri import ASTModel
from sklearn import metrics
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_path", type=str, default='', help="the root path of the experiment")
parser.add_argument("--exp_path2", type=str, default='', help="the root path of the experiment")
args = parser.parse_args()

val_audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': "resp",
                  'mode': 'evaluation', 'mean': -10.36892, 'std': 5.984859, 'noise': False}

val_loader = torch.utils.data.DataLoader(
    dataloader.AudioDataset("./finetune/Resp_Coarse/data/datafiles/test.json", label_csv="./finetune/Resp_Coarse/data/resp_label.csv", audio_conf=val_audio_conf),
    batch_size=12, shuffle=False, num_workers=16, pin_memory=False)

audio_model = ASTModel(label_dim=2, fshape=16, tshape=16, fstride=10, tstride=10,
                       input_fdim=128, input_tdim=1024, model_size="base", pretrain_stage=False,
                       load_pretrained_mdl_path="/homes/anyan/respire-sound-detection/ssast/src/finetune/Resp_Coarse/SSAST-Base-Patch-400.pth", freeze_base=False, task="ft_avgmaxtok")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load(f"{args.exp_path}/fold1/models/best_audio_model.pth", map_location=device)
audio_model = torch.nn.DataParallel(audio_model)
audio_model.load_state_dict(sd, strict=False)
audio_model = audio_model.to(device)
audio_model.eval()
classes_num = 2
A_predictions = []
A_targets = []
A_embeddings = []
A_name = []
with torch.no_grad():
    epoch = 0
    for i, (wav_name, audio_input, labels) in enumerate(val_loader):
        audio_input = audio_input.to(device)

        audio_output = audio_model(audio_input, "ft_avgmaxtok")
        audio_output_ = torch.sigmoid(audio_output['x'])
        predictions = audio_output_.to('cpu').detach()
        
        A_predictions.append(predictions)
        A_targets.append(labels)
        for wav in wav_name:
            A_name.append(wav)
        print(i)
        epoch += 1
        
    audio_output = torch.cat(A_predictions)
    target = torch.cat(A_targets)
    out = {'name': A_name, 'predict': audio_output.cpu().numpy()}
    with open(f"{args.exp_path}/Predict_test.pkl", "wb") as f:
        pickle.dump(out, f)
        
with open(f"{args.exp_path}/Predict_test.pkl", "rb") as f:
    out = pickle.load(f)

with open(f"{args.exp_path2}/evaluate/predict.pickle", "rb") as f:
    panns = pickle.load(f)

out['predict'] = np.argmax(out['predict'], axis = 1)
lbl_to_idx = {'Coarse': 0, 'Normal': 1, 'Wheeze': 2}
target = []
prediction = []
for i in range(len(panns['audio'])):
    target.append(lbl_to_idx[panns['audio'][i].split('.')[0]])
    if panns['predict'][i] == 1:
        prediction.append(2)
    else:
        for j in range(len(out['name'])):
            if panns['audio'][i].split('/')[-1] == out['name'][j].split('/')[-1]:
                prediction.append(out['predict'][j])
                break

cm = metrics.confusion_matrix(target, prediction)
print(cm)
sensitivity = (cm[0][0]+cm[2][2])/(np.sum(cm[0])+np.sum(cm[2]))
specificity = cm[1][1]/np.sum(cm[1])
accuracy = np.sum(np.array(target) == np.array(prediction)) / len(target)
print('accuracy: ', accuracy)
print('sensitivity: ', sensitivity)
print('specificity: ', specificity)
print('ICBHI score: ', (sensitivity + specificity) / 2)
      
        