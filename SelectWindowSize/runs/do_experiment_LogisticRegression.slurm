#!/usr/bin/env bash

python3 /cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental-workload-classifiers/SelectWindowSize/LogisticRegression.py \
    --data_dir $data_dir \
    --SelectWindowSize_testset_dir $SelectWindowSize_testset_dir \
    --window_size $window_size \
    --result_save_rootdir $result_save_rootdir \
    --SubjectId_of_interest $SubjectId_of_interest \
    --classification_task $classification_task
    
