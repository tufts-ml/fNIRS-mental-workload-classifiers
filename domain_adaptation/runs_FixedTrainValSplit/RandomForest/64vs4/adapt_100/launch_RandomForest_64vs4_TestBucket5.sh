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
export data_dir='/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/data/bpf_Leon/Visual/size_30sec_150ts_stride_3ts/'
export window_size=150
export classification_task='binary'
export scenario='64vs4'
export bucket='TestBucket5'
export setting="64vs4_TestBucket5"
export adapt_on='train_100'
export result_save_rootdir="/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/experiments/FixedTrainValSplit_domain_adaptation/RandomForest/binary/$adapt_on/$scenario/$bucket" 


if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < /cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/domain_adaptation/runs_FixedTrainValSplit/do_experiment_RandomForest.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash /cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/domain_adaptation/runs_FixedTrainValSplit/do_experiment_RandomForest.slurm
fi

