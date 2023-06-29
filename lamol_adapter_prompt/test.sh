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

python test_adapter_from_scratch.py --data_dir data --model_dir model_conf2 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 12 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0

python test_adapter_from_scratch.py --data_dir data --model_dir model_conf3 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 12 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0

python test_adapter_from_scratch.py --data_dir data --model_dir model_prefix --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 12 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_adap --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 12 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0 > Testlogs_adapter_3og_preadap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_conf4 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 12 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0 > Testlogs_conf4_3og_preadap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_cm1 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 20 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0 > Testlogs_cm1_3og_preadap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_am1 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 20 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0 > Testlogs_am1_3og_preadap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_pm1 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 20 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0 > Testlogs_pm1_3og_preadap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_mam1 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 20 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0 > Testlogs_mam1_3og_preadap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_c10 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 20 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0 > Testlogs_c10_3og_preadap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_c20 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 20 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0 > Testlogs_c20_3og_preadap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_c40 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 20 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0 > Testlogs_c40_3og_preadap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_cm2 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 20 --gen_lm_sample_percentage 0.0 --tasks wikisql ag amazon sst srl woz.en --lm_lambda 0.0 > Testlogs_cm2_6f_preadap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_am2 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 20 --gen_lm_sample_percentage 0.0 --tasks wikisql ag amazon sst srl woz.en --lm_lambda 0.0 > Testlogs_am2_6f_preadap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_pm2 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 20 --gen_lm_sample_percentage 0.0 --tasks wikisql ag amazon sst srl woz.en --lm_lambda 0.0 > Testlogs_pm2_6f_preadap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_am3 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 20 --gen_lm_sample_percentage 0.0 --tasks sst srl zre woz.en --lm_lambda 0.0 > Testlogs_am3_4zre_preadap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_cm3 --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 20 --gen_lm_sample_percentage 0.0 --tasks sst srl zre woz.en --lm_lambda 0.0 > Testlogs_cm3_4zre_preadap.out

nohup python test_adapter_from_scratch.py --data_dir data --model_dir model_wandbtest --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 25 --n_train_epochs 3 --gen_lm_sample_percentage 0.0 --tasks sst srl woz.en --lm_lambda 0.0 > Testlogs_wandbtest_3og_preadap.out