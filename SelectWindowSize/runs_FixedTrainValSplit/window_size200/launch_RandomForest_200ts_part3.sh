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
export data_dir='/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/data/bpf_Leon/Visual/size_40sec_200ts_stride_3ts/'
export SelectWindowSize_testset_dir='/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/data/bpf_UsedForSelectingWindowSize/Visual/size_40sec_200ts_stride_3ts/'
export window_size=200
export result_save_rootdir='/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/experiments/FixedTrainValSplit_SelectWindowSize/RandomForest/binary/window_size200'
export classification_task='binary'
for SubjectId_of_interest in 81 29 7 74 86 25 79 76 18 48 95 1 8 61 51 70 17 64 62 49 9 72

do
    export SubjectId_of_interest=$SubjectId_of_interest

    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < /cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/SelectWindowSize/runs_FixedTrainValSplit/do_experiment_RandomForest_FixedTrainValSplit.slurm
    
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash /cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/SelectWindowSize/runs_FixedTrainValSplit/do_experiment_RandomForest_FixedTrainValSplit.slurm
    fi

done

