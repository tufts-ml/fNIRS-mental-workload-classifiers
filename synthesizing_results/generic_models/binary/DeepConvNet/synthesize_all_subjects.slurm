#!/usr/bin/env bash


# Pass along all ENV variables as arguments to my Python script
python /cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental-workload-classifiers/fNIRS-mental_workload-classifiers/generic_models/synthesize_all_subjects.py\
    --experiment_dir $experiment_dir \
    