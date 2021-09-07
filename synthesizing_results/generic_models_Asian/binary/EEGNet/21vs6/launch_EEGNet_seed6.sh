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


for SubjectId_of_interest in 22 70 78 28 60 58 86 80 48 31 21 14 71 73 61 68 25 35
do
    export experiment_dir="/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/experiments/FixedTrainValSplit_generic_models/AsianSubset/EEGNet/binary/21vs6/seed6/$SubjectId_of_interest"
    
    echo "Current experiment_dir is $experiment_dir"
    
    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < /cluster/tufts/hugheslab/zhuang12/general_utilities/synthesize_hyperparameter_search_results/NuripsDataSet_FixedTrainValSplit/generic_models_Asian/binary/EEGNet/synthesize_hypersearch_EEGNet_for_a_subject.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash /cluster/tufts/hugheslab/zhuang12/general_utilities/synthesize_hyperparameter_search_results/NuripsDataSet_FixedTrainValSplit/generic_models_Asian/binary/EEGNet/synthesize_hypersearch_EEGNet_for_a_subject.slurm
    fi
    
done
