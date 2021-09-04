#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=$gpu_idx python3 /cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental-workload-classifiers/subgroup_analysis/AsianSubset/RandomForest.py \
    --data_dir $data_dir \
    --window_size $window_size \
    --classification_task $classification_task\
    --result_save_rootdir $result_save_rootdir \
    --setting $setting \

    
