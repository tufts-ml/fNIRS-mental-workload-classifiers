#!/usr/bin/env bash

python3 /cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/generic_finetuning_models/EEGNet.py \
    --data_dir $data_dir \
    --window_size $window_size \
    --classification_task $classification_task\
    --result_save_rootdir $result_save_rootdir \
    --setting $setting \
    --restore_file $restore_file\
    --n_epoch $n_epoch\
    --adapt_on $adapt_on\

    
