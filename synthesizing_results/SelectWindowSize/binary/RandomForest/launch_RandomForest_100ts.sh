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


for SubjectId_of_interest in 4 41 69 3 15 52 42 38 34 66 35 24 40 26 16 80 27 73 20 12 11 67 94 44 92 75 5 59 71 28 47 85 68 55 60 91 84 21 37 56 36 10 83 93 81 29 7 74 86 25 79 76 18 48 95 1 8 61 51 70 17 64 62 49 9 72 45 43 63 14 19 2 57 82 53 54 46 97 22 50 32 78 30 31 23 58 65 13
do
    export experiment_dir="/cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental-workload-classifiers/experiments/SelectWindowSize/RandomForest/binary/window_size100/$SubjectId_of_interest"
    
    echo "Current experiment_dir is $experiment_dir"
    
    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < /cluster/tufts/hugheslab/zhuang12/general_utilities/synthesize_hyperparameter_search_results/NuripsDataSet_FixedTrainValSplit/SelectWindowSize/binary/RandomForest/synthesize_hypersearch_RF_for_a_subject.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash /cluster/tufts/hugheslab/zhuang12/general_utilities/synthesize_hyperparameter_search_results/NuripsDataSet_FixedTrainValSplit/SelectWindowSize/binary/RandomForest/synthesize_hypersearch_RF_for_a_subject.slurm
    fi
    
done
