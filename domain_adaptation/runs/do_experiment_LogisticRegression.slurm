#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=$gpu_idx python3 YOUR_PATH/fNIRS-mental-workload-classifiers/domain_adaptation/run_GenericLogisticRegression_with_CORAL.py \
    --data_dir $data_dir \
    --window_size $window_size \
    --classification_task $classification_task\
    --result_save_rootdir $result_save_rootdir \
    --setting $setting \
    --adapt_on $adapt_on \

    
