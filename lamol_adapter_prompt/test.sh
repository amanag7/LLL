#!/bin/bash -i

source ./env

python test.py \
  --data_dir $DATA_DIR \
  --model_dir_root $MODEL_ROOT_DIR \
  "$@"

nohup python test_adapter_from_scratch.py --data_dir data --model_dir models --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 12 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0 > Testlogs_3og_adap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir new_models --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 12 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0 > Testlogs_3og_preadap.out

python test_adapter_from_scratch.py --data_dir data --model_dir new_models --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 2 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0

python test_adapter_from_scratch.py --data_dir data --model_dir model_conf1 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 12 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0