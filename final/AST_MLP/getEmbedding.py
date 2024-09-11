import os
import pickle
import sys
import torch
from torch import nn
import dataloader
from ast_models_tri import ASTModel
import numpy as np

val_audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': "resp",
                  'mode': 'evaluation', 'mean': -10.377555, 'std': 5.9955864, 'noise': False}
audio_model = ASTModel(label_dim=3, fshape=16, tshape=16, fstride=10, tstride=10,
                    input_fdim=128, input_tdim=1024, model_size="base", pretrain_stage=False,
                    load_pretrained_mdl_path="./models/Pretrain.pth", freeze_base=False, task="ft_avgmaxtok")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load(f"./Resp/exp/avgmax_tri005_mix05_bal-resp-f10-16-t10-16-b6-lr1e-5-ft_avgmaxtok-base--SSAST-Base-Patch-400-1x-noiseTrue/fold{j}/models/best_audio_model.pth", map_location=device)
audio_model = torch.nn.DataParallel(audio_model)
audio_model.load_state_dict(sd, strict=False)
audio_model = audio_model.to(device)
audio_model.eval()
for j in range(1,6):
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(f"./Resp/data/datafiles/Resp_train_data_{j}.json", label_csv="./Resp/data/resp_label.csv", audio_conf=val_audio_conf),
        batch_size=12, shuffle=False, num_workers=16, pin_memory=False)

    classes_num = 3
    A_predictions = []
    A_targets = []
    A_embeddings = []
    with torch.no_grad():
        epoch = 0
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)
            audio_output = audio_model(audio_input, "ft_avgmaxtok")
            audio_output_ = torch.sigmoid(audio_output['x'])
            predictions = audio_output_.to('cpu').detach()
            
            A_predictions.append(predictions)
            A_targets.append(labels)
            A_embeddings.append(audio_output['embedding'])
            print(epoch)
            epoch += 1
            
        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        embeddings = torch.cat(A_embeddings)
        with open(f"../ML/embeddings_train_{j}.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        with open(f"../ML/labels_train_{j}.pkl", "wb") as f:
            pickle.dump(target, f)
            
            
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(f"./Resp/data/datafiles/Resp_eval_data_{j}.json", label_csv="./Resp/data/resp_label.csv", audio_conf=val_audio_conf),
        batch_size=12, shuffle=False, num_workers=16, pin_memory=False)

    audio_model.eval()
    classes_num = 3
    A_predictions = []
    A_targets = []
    A_embeddings = []
    with torch.no_grad():
        epoch = 0
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)
            audio_output = audio_model(audio_input, "ft_avgmaxtok")
            audio_output_ = torch.sigmoid(audio_output['x'])
            predictions = audio_output_.to('cpu').detach()
            
            A_predictions.append(predictions)
            A_targets.append(labels)
            A_embeddings.append(audio_output['embedding'])
            print(epoch)
            epoch += 1
            
        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        embeddings = torch.cat(A_embeddings)
        with open(f"../ML/embeddings_val_{j}.pkl", "wb") as f:
            pickle.dump(embeddings, f)
        with open(f"../ML/labels_val_{j}.pkl", "wb") as f:
            pickle.dump(target, f)
