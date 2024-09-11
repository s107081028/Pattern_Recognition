export TORCH_HOME=../../pretrained_models
mkdir exp

dataset=AudioSet
dataset_mean=-10.315503
dataset_std=5.979837
target_length=1024
noise=True

bal='bal'
lr=1e-5
freqm=24
timem=96
mixup=0.5
epoch=40
batch_size=6
fshape=16
tshape=16
fstride=10
tstride=10

task=pretrain_joint
model_size=base
head_lr=1

base_exp_dir=./exp/${task}-mix${mixup}-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}

for((fold=1;fold<=5;fold++));
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}

  tr_data=./data/datafiles/train_data_${fold}.json
  te_data=./data/datafiles/eval_data_${fold}.json

  CUDA_VISIBLE_DEVICES=1, CUDA_CACHE_DISABLE=1 python -W ignore ../../run_pretrain.py --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --label-csv ./data/label.csv \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup False --task ${task} \
  --model_size ${model_size} --adaptschedule False \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
  --num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
  --lrscheduler_start 6 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics mAP
done

python ./get_result.py --exp_path ${base_exp_dir}
python ./cum_ensemble_result.py --exp_path ${base_exp_dir}
