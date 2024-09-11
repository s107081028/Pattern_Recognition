mkdir exp

pretrain_exp=
pretrain_model=Pretrain
pretrain_path=./${pretrain_exp}/${pretrain_model}.pth

dataset=resp
dataset_mean=-10.36892
dataset_std=5.984859
target_length=1024
noise=True

bal='bal'
# bal=None
lr=1e-4
freqm=24
timem=96
mixup=0.5
epoch=40
batch_size=8
fshape=16
tshape=16
fstride=10
tstride=10

task=ft_avgtok
model_size=base
head_lr=1

base_exp_dir=./exp/bal_tri-${dataset}-f${fstride}-${fshape}-t${tstride}-${tshape}-b${batch_size}-lr${lr}-${task}-${model_size}-${pretrain_exp}-${pretrain_model}-${head_lr}x-noise${noise}

for((fold=1;fold<=5;fold++));
do
  echo 'now process fold'${fold}

  exp_dir=${base_exp_dir}/fold${fold}

  tr_data=./data/datafiles/Resp_train_data_${fold}.json
  te_data=./data/datafiles/Resp_eval_data_${fold}.json

  CUDA_VISIBLE_DEVICES=0, CUDA_CACHE_DISABLE=1 python -W ignore ../run_finetune.py --dataset ${dataset} \
  --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
  --label-csv ./data/resp_label.csv --n_class 3 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --task ${task} \
  --model_size ${model_size} --pretrained_mdl_path ${pretrain_path} \
  --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
  --num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
  --lrscheduler_start 6 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE_TRI --metrics acc
done

python ./get_resp_result.py --exp_path ${base_exp_dir}
python ./cum_ensemble_result.py --exp_path ${base_exp_dir} --json_path ./data/datafiles
