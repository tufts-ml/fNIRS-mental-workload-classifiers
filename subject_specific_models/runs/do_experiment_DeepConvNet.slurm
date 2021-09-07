#!/usr/bin/env bash

python3 /cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental-workload-classifiers/subject_specific_models/DeepConvNet.py \
    --gpu_idx $gpu_idx \
    --data_dir $data_dir \
    --window_size $window_size \
    --result_save_rootdir $result_save_rootdir \
    --classification_task $classification_task\
    --restore_file $restore_file\
    --n_epoch $n_epoch\
    --SubjectId_of_interest $SubjectId_of_interest \
    
