#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

export gpu_idx=0
export data_dir='YOUR_PATH/fNIRS-mental-workload-classifiers/data/chunk_data/size_30sec_150ts_stride_3ts/'
export window_size=150
export classification_task='binary'
export scenario='64vs4'
export bucket='TestBucket14'
export setting="64vs4_TestBucket14"
export n_epoch=300
export restore_file='YOUR_PATH/fNIRS-mental-workload-classifiers/experiments/generic_models/DeepConvNet/binary/64vs4/TestBucket14/46/lr0.01_dropout0.25/checkpoint/best_model.statedict'
export adapt_on='train_100'
export result_save_rootdir="YOUR_PATH/fNIRS-mental-workload-classifiers/experiments/generic_finetuning_models/DeepConvNet/binary/$adapt_on/$scenario/$bucket" 



if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < YOUR_PATH/fNIRS-mental-workload-classifiers/generic_finetuning_models/runs/do_experiment_DeepConvNet.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash YOUR_PATH/fNIRS-mental-workload-classifiers/generic_finetuning_models/runs/do_experiment_DeepConvNet.slurm
fi

