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
export scenario='21vs6'
export setting="seed2"
export result_save_rootdir="/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/experiments/FixedTrainValSplit_generic_models/AsianSubset/EEGNet/binary/$scenario/$setting" 
export n_epoch=600
export restore_file='None'

if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch < /cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/generic_models/AsianSubset/runs_FixedTrainValSplit/do_experiment_EEGNet.slurm

elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to just run interactively
    bash /cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/generic_models/AsianSubset/runs_FixedTrainValSplit/do_experiment_EEGNet.slurm
fi

