# Sample bash
# python ./cum_ensemble_result.py --exp_path /homes/anyan/respire-sound-detection/ssast/src/finetune/Relabel_Resp/exp/test01-resp-f10-16-t10-16-b16-lr1e-4-ft_avgtok-base--SSAST-Base-Patch-400-1x-noiseTrue

import argparse
import numpy as np
import pickle
import json
from sklearn import metrics
import logging

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_path", type=str, default='', help="the root path of the experiment")
parser.add_argument("--epochs", type=int, default=40, help="the root path of the data json")

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(filename=args.exp_path+'/result.log', filemode='w', level=logging.DEBUG)
    json_path = './data/datafiles'
    
    ground_truth_dict = {}
    prediction_dict = {}
    for fold in range(1, 6):
        prediction = np.loadtxt(args.exp_path+'/fold' + str(fold) + '/predictions/cum_predictions.csv', delimiter=',')
        ground_truth = json.load(open(json_path+'/Resp_eval_data_'+str(fold)+'.json'))
        ground_truth = ground_truth['data']
        for i in range(len(ground_truth)):
            if ground_truth[i]["wav"] not in ground_truth_dict.keys(): 
                ground_truth_dict[ground_truth[i]["wav"]] = int(ground_truth[i]["labels"][-1])
                prediction_dict[ground_truth[i]["wav"]] = prediction[i]
            else:
                if ground_truth_dict[ground_truth[i]["wav"]] != int(ground_truth[i]["labels"][-1]):
                    assert("Different Ground Truth!")
                prediction_dict[ground_truth[i]["wav"]] += prediction[i]
    
    predictions = []
    targets = []
    for key in ground_truth_dict.keys():
        targets.append(ground_truth_dict[key])
        predictions.append(np.argmax(prediction_dict[key]))
    
    target_names = ['Coarse', 'Normal','Wheeze']
    print(metrics.classification_report(targets, predictions, target_names=target_names, digits = 3))
    logging.info(f'\n{metrics.classification_report(targets, predictions, target_names=target_names, digits = 3)}')
    cm = metrics.confusion_matrix(targets, predictions)
    specificity = cm[1][1]/np.sum(cm[1])
    sensitivity = (cm[0][0]+cm[2][2])/(np.sum(cm[0])+np.sum(cm[2]))
    print(cm)
    logging.info(f'\n{cm}')
    print('Sensitivity: {}'.format(sensitivity))
    logging.info('Sensitivity:{}'.format(sensitivity))
    print('Specificity:{}'.format(specificity))
    logging.info('Specificity: {}'.format(specificity))
    