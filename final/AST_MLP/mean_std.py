import torch
import numpy as np
from dataloader import AudioDataset

audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 24, 'timem': 192, 'mixup': 0.5, 'skip_norm': True, 'mode': 'train', 'dataset': 'audioset'}
train_loader = torch.utils.data.DataLoader(
    AudioDataset('Resp/data/datafiles/Resp_data.json', label_csv='Resp/data/resp_label.csv',
                                audio_conf=audio_conf), batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)
mean=[]
std=[]
for i, (audio_input, labels) in enumerate(train_loader):
    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
    print(cur_mean, cur_std)
print(np.mean(mean), np.mean(std))