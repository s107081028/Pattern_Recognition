import argparse
import os
import ast
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
import dataloader
import dataloader_aug
from models.ast_models_tri import ASTModel
from traintest_mask import trainmask

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default=None, help="training data json")
parser.add_argument("--data-val", type=str, default=None, help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")

parser.add_argument("--dataset", type=str, help="the dataset used for training")
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--num_mel_bins", type=int, default=128, help="number of input mel bins")
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
parser.add_argument("--fstride", type=int, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument("--fshape", type=int, help="shape of patch on the frequency dimension")
parser.add_argument("--tshape", type=int, help="shape of patch on the time dimension")

parser.add_argument('--model_size', help='the size of AST models', type=str, default='base384')
parser.add_argument("--task", type=str, default='pretrain_joint', help="pretraining or fine-tuning task", choices=["pretrain_mpc", "pretrain_mpg", "pretrain_joint"])

parser.add_argument('--pretrain_stage', help='True for self-supervised pretraining stage, False for fine-tuning stage', type=ast.literal_eval, default='True')
parser.add_argument('--mask_patch', help='how many patches to mask (used only for ssl pretraining)', type=int, default=400)
parser.add_argument("--cluster_factor", type=int, default=3, help="mask clutering factor")
parser.add_argument("--epoch_iter", type=int, default=2000, help="for pretraining, how many iterations to verify and save models")

args = parser.parse_args()
audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset,
              'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise}
val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}

samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
if args.aug:
    train_loader = torch.utils.data.DataLoader(
        dataloader_aug.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False, drop_last=True)
else:    
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudioDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=False)

cluster = (args.num_mel_bins != args.fshape)
if cluster == True:
    print('The num_mel_bins {:d} and fshape {:d} are different, not masking a typical time frame, using cluster masking.'.format(args.num_mel_bins, args.fshape))
else:
    print('The num_mel_bins {:d} and fshape {:d} are same, masking a typical time frame, not using cluster masking.'.format(args.num_mel_bins, args.fshape))

audio_model = ASTModel(fshape=args.fshape, tshape=args.tshape, fstride=args.fshape, tstride=args.tshape,
                    input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=True)

if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)

if os.path.exists("%s/models" % args.exp_dir) == False:
    os.makedirs("%s/models" % args.exp_dir)

trainmask(audio_model, train_loader, val_loader, args)