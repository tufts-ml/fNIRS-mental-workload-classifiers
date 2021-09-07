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
export data_dir='/cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental-workload-classifiers/data/chunk_data/size_30sec_150ts_stride_3ts/'
export window_size=150
export classification_task='binary'
export scenario='16vs4'
export bucket='TestBucket4'
export setting="16vs4_TestBucket4"
export result_save_rootdir="/cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental-workload-classifiers/experiments/generic_models/DeepConvNet/binary/$scenario/$bucket"
export n_epoch=600
export restore_file='None'

if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < /cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental-workload-classifiers/generic_models/runs/do_experiment_DeepConvNet.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash /cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental-workload-classifiers/generic_models/runs/do_experiment_DeepConvNet.slurm
fi

