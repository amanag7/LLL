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

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root new_models --seq_train_type lll --model_name gpt2 --n_gpus 6 --n_workers 75 --fp32 --n_train_epochs 2 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_testing_3og_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_conf1 --seq_train_type lll --model_name gpt2 --n_gpus 6 --n_workers 75 --fp32 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_conf1_3og_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_conf2 --seq_train_type lll --model_name gpt2 --n_gpus 6 --n_workers 75 --fp32 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_conf2_3og_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_conf3 --seq_train_type lll --model_name gpt2 --n_gpus 6 --n_workers 75 --fp32 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_conf3_3og_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_prefix --seq_train_type lll --model_name gpt2 --n_gpus 6 --n_workers 75 --fp32 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_prefix_3og_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_adap --seq_train_type lll --model_name gpt2 --n_gpus 6 --n_workers 75 --fp32 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_adapter_3og_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_conf4 --seq_train_type lll --model_name gpt2 --n_gpus 6 --n_workers 75 --fp32 --n_train_epochs 12 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_conf4_3og_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_cm1 --seq_train_type lll --model_name gpt2 --n_gpus 4 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_cm1_3og_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_am1 --seq_train_type lll --model_name gpt2 --n_gpus 4 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_am1_3og_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_pm1 --seq_train_type lll --model_name gpt2 --n_gpus 4 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_pm1_3og_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_mam1 --seq_train_type lll --model_name gpt2 --n_gpus 4 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_mam1_3og_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_cm2 --seq_train_type lll --model_name gpt2 --n_gpus 2 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.00 --tasks wikisql ag amazon sst srl woz.en --lm_lambda 0.0 > Trainlogs_cm2_6f_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_am2 --seq_train_type lll --model_name gpt2 --n_gpus 2 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.00 --tasks wikisql ag amazon sst srl woz.en --lm_lambda 0.0 > Trainlogs_am2_6f_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_pm2 --seq_train_type lll --model_name gpt2 --n_gpus 2 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.00 --tasks wikisql ag amazon sst srl woz.en --lm_lambda 0.0 > Trainlogs_pm2_6f_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_c10 --seq_train_type lll --model_name gpt2 --n_gpus 2 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_c10_3og_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_c20 --seq_train_type lll --model_name gpt2 --n_gpus 2 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_c20_3og_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_c40 --seq_train_type lll --model_name gpt2 --n_gpus 2 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0 > Trainlogs_c40_3og_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_cm3 --seq_train_type lll --model_name gpt2 --n_gpus 3 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.00 --tasks sst srl zre woz.en --lm_lambda 0.0 > Trainlogs_cm3_4zre_preadap.out

nohup python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_am3 --seq_train_type lll --model_name gpt2 --n_gpus 3 --n_workers 75 --fp32 --n_train_epochs 20 --gen_lm_sample_percentage 0.00 --tasks sst srl zre woz.en --lm_lambda 0.0 > Trainlogs_am3_4zre_preadap.out

python train_experiment_prefix_adapter.py --data_dir data --model_dir_root model_wandbtest --seq_train_type lll --model_name gpt2 --n_gpus 1 --n_workers 75 --fp32 --n_train_epochs 3 --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0