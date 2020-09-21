#!/usr/bin/env bash

if [ ! -d ./datasets ];then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/clue_glue_superglue_benchmark/clue_datasets.tgz
  tar -zxf clue_datasets.tgz
  mkdir datasets
  mv clue_datasets/* datasets
  rm -rf *_datasets
  rm *.tgz
fi
export CUDA_VISIBLE_DEVICES="0"
#CLUE---> AFQMC, CMNLI, CSL, IFLYTEK, TNEWS
task_name=WSC

python3 main_finetune.py --workerGPU=1 \
  --mode="train_and_evaluate_on_the_fly" \
  --task_name=${task_name}  \
  --train_input_fp=datasets/${task_name}/train.csv  \
  --eval_input_fp=datasets/${task_name}/dev.csv  \
  --pretrain_model_name_or_path=google-bert-base-zh  \
  --train_batch_size=16  \
  --num_epochs=2.5  \
  --model_dir=${task_name}_model_dir  \
  --learning_rate=1e-5  \


predict_checkpoint_path=${task_name}_model_dir/model.ckpt-77
python3 main_finetune.py --workerGPU=1 \
  --mode="predict_on_the_fly" \
  --predict_checkpoint_path=$predict_checkpoint_path \
  --task_name=${task_name}  \
  --train_input_fp=datasets/${task_name}/train.csv  \
  --eval_input_fp=datasets/${task_name}/dev.csv  \
  --predict_input_fp=datasets/${task_name}/dev.csv  \
  --pretrain_model_name_or_path=google-bert-base-zh  \
  --train_batch_size=16  \
  --num_epochs=2.5  \
  --model_dir=${task_name}_model_dir  \
  --learning_rate=1e-5  \


