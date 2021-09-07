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


# for experiment in 64vs4 16vs4 4vs4
for experiment in 64vs4
do
    export experiment_dir="/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/experiments/FixedTrainValSplit_generic_finetuning_models/EEGNet/binary/train_100/$experiment"
    
    echo "Current experiment_dir is $experiment_dir"
    
    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < /cluster/tufts/hugheslab/zhuang12/general_utilities/synthesize_hyperparameter_search_results/NuripsDataSet_FixedTrainValSplit/generic_finetuning_models/binary/EEGNet/train_100/synthesize_all_subjects.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash /cluster/tufts/hugheslab/zhuang12/general_utilities/synthesize_hyperparameter_search_results/NuripsDataSet_FixedTrainValSplit/generic_finetuning_models/binary/EEGNet/train_100/synthesize_all_subjects.slurm
    fi
    
done