#!/bin/bash -i

source ./env

python train.py \
  --data_dir $DATA_DIR \
  --model_dir_root $MODEL_ROOT_DIR \
  "$@"

nohup python train_experiment_adapter_from_scratch_mam_config.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 8 --n_workers 75 --fp32 --n_train_epochs 4 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_3decaNLP_mam_config.out

nohup python train_experiment_adapter_prompts.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 8 --n_workers 75 --fp32 --n_train_epochs 2 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_3og_prompts.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root new_models --seq_train_type lll --model_name gpt2 --n_gpus 8 --n_workers 75 --fp32 --n_train_epochs 2 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_3og_preadap.out

nohup python train_experiment_adapter_from_scratch_mam_config.py --data_dir data --model_dir_root models --seq_train_type lll --model_name gpt2 --n_gpus 2 --n_workers 75 --fp32 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_3og_adap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root new_models --seq_train_type lll --model_name gpt2 --n_gpus 2 --n_workers 75 --fp32 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_3og_preadap.out