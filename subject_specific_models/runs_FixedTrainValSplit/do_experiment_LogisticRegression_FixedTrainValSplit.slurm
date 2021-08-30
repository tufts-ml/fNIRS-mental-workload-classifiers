#!/usr/bin/env bash

python3 /cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/subject_specific_models/LogisticRegression_FixedTrainValSplit.py \
    --data_dir $data_dir \
    --window_size $window_size \
    --result_save_rootdir $result_save_rootdir \
    --SubjectId_of_interest $SubjectId_of_interest \
    --classification_task $classification_task
    
