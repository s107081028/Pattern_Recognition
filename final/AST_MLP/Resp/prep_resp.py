import numpy as np
import json
import os
import random
import csv
import shutil

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

base_dir = './data/'
audio16k_list = get_immediate_files('./data/audios')
print(audio16k_list[:10])

label_map = {'Coarse': 0, 'Normal': 1, 'Wheeze': 2}
with open(base_dir + 'resp_label.csv', 'w', newline='') as csvfile:
    fieldnames = ['index', 'mid', 'display_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(0, len(label_map.keys())):
        index = i
        mid = '/m/07rwj' + str(index).zfill(2)
        display_name = label_map[list(label_map.keys())[i]]
        writer.writerow({'index': index, 'mid': mid, 'display_name': display_name})

# # fix bug: generate an empty directory to save json files
if os.path.exists('./data/datafiles') == False:
    os.mkdir('./data/datafiles')

for fold in [1,2,3,4,5]:
    base_path = os.path.abspath(os.getcwd()) + "/data/audio_16k/"
    train_wav_list = []
    eval_wav_list = []
    weight_list = [0] * len(label_map)
    for i in range(0, len(audio16k_list)):
        cur_label = label_map[audio16k_list[i].split('.')[0]]
        cur_path = audio16k_list[i]
        cur_fold = int(audio16k_list[i][-5])
        # /m/07rwj is just a dummy prefix
        cur_dict = {"wav": base_path + cur_path, "labels": '/m/07rwj' + str(cur_label).zfill(2)}
        if cur_fold == fold:
            eval_wav_list.append(cur_dict)
        else:
            weight_list[cur_label] += 1
            train_wav_list.append(cur_dict)

    total = sum(weight_list)
    train_weight_list = [0] * len(train_wav_list)
    for i in range(len(weight_list)):
        weight_list[i] = total / weight_list[i] 
    
    total = sum(weight_list)
    for i in range(len(weight_list)):
        weight_list[i] = weight_list[i] / total
    
    weight_list = [0.1, 0.05, 0.9]
    for i in range(len(train_wav_list)):
        train_weight_list[i] = weight_list[int(train_wav_list[i]["labels"][-1])]

    print(f'fold {fold}: {len(train_wav_list)} training samples, {len(eval_wav_list)} test samples, weights are {weight_list}')

    with open('./data/datafiles/Resp_train_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)

    with open('./data/datafiles/Resp_eval_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)
        
    np.savetxt('./data/datafiles/Resp_train_data_'+ str(fold) + '_weight.csv', train_weight_list, delimiter=',')

print('Finished Relabel Resp Preparation')

# All data for calculating mean std
base_path = os.path.abspath(os.getcwd()) + "/data/audio_16k/"
wav_list = []
for i in range(0, len(audio16k_list)):
    cur_label = label_map[audio16k_list[i].split('.')[0]]
    cur_path = audio16k_list[i]
    cur_fold = int(audio16k_list[i][-5])
    # /m/07rwj is just a dummy prefix
    cur_dict = {"wav": base_path + cur_path, "labels": '/m/07rwj' + str(cur_label).zfill(2)}
    wav_list.append(cur_dict)

print('{:d} samples'.format(len(wav_list)))

with open('./data/datafiles/Resp_data.json', 'w') as f:
    json.dump({'data': wav_list}, f, indent=1)
