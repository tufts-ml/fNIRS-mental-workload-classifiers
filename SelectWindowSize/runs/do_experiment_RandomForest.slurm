#!/usr/bin/env bash

python3 YOUR_PATH/fNIRS-mental-workload-classifiers/SelectWindowSize/run_RandomForest.py \
    --data_dir $data_dir \
    --SelectWindowSize_testset_dir $SelectWindowSize_testset_dir \
    --window_size $window_size \
    --result_save_rootdir $result_save_rootdir \
    --SubjectId_of_interest $SubjectId_of_interest \
    --classification_task $classification_task
    
