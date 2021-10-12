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


for SubjectId_of_interest in 22 70 78 28 60 58 69 85 34 31 86 32 51 25 44 65 52 56
do
    export experiment_dir="YOUR_PATH/fNIRS-mental-workload-classifiers/experiments/generic_models/WhiteSubset/EEGNet/binary/21vs6/random_partition4/$SubjectId_of_interest"
    
    echo "Current experiment_dir is $experiment_dir"
    
    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < YOUR_PATH/fNIRS-mental_workload-classifiers/synthesizing_results/subgroup_White/binary/EEGNet/synthesize_hypersearch_EEGNet_for_a_subject.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash YOUR_PATH/fNIRS-mental_workload-classifiers/synthesizing_results/subgroup_White/binary/EEGNet/synthesize_hypersearch_EEGNet_for_a_subject.slurm
    fi
    
done
