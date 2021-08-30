#NOTE: run this script with the bpf data, use 5050 paradigm

import os
import sys
import numpy as np
import argparse

from easydict import EasyDict as edict
from tqdm import trange
# from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, '/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/helpers/')
import models
import brain_data
from utils import seed_everything, featurize, makedir_if_not_exist, plot_confusion_matrix, save_pickle, write_performance_info_FixedTrainValSplit

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--data_dir', default='../data/bpf_Leon/Visual/size_2sec_10ts_stride_3ts/', help='folder to the train data')
parser.add_argument('--SelectWindowSize_testset_dir', default='../data/bpf_UsedForSelectingWindowSize/Visual/size_2sec_10ts_stride_3ts', help='folder to the test data')
parser.add_argument('--window_size', default=200, type=int, help='window size')
parser.add_argument('--result_save_rootdir', default='./experiments', help='folder to the result')
parser.add_argument('--SubjectId_of_interest', default='1', help='which subject of interest')
parser.add_argument('--classification_task', default='four_class', help='binary or four-class classification')

def train_classifier(args_dict):
    
    #parse args:
    data_dir = args_dict.data_dir
    SelectWindowSize_testset_dir = args_dict.SelectWindowSize_testset_dir
    window_size = args_dict.window_size
    result_save_rootdir = args_dict.result_save_rootdir
    SubjectId_of_interest = args_dict.SubjectId_of_interest
    classification_task = args_dict.classification_task
    
    #load this subject's data
    sub_file = 'sub_{}.csv'.format(SubjectId_of_interest)
    
    if window_size == 10:
        num_chunk_this_window_size = 2224
    elif window_size == 25:
        num_chunk_this_window_size = 2144
    elif window_size == 50:
        num_chunk_this_window_size = 2016
    elif window_size == 100:
        num_chunk_this_window_size = 1744
    elif window_size == 150:
        num_chunk_this_window_size = 1488
    elif window_size == 200:
        num_chunk_this_window_size = 1216
    else:
        raise NameError('not supported window size')
        
        
    if classification_task == 'four_class':
        data_loading_function = brain_data.read_subject_csv
        data_loading_function_testset = brain_data.read_subject_csv_SelectWindowSize
        confusion_matrix_figure_labels = ['0back', '1back', '2back', '3back']
        
    elif classification_task == 'binary':
        data_loading_function = brain_data.read_subject_csv_binary
        data_loading_function_testset = brain_data.read_subject_csv_binary_SelectWindowSize
        confusion_matrix_figure_labels = ['0back', '2back']
        
    else:
        raise NameError('not supported classification type')
        
    
    #load the subject's data
    sub_feature_array, sub_label_array = data_loading_function(os.path.join(data_dir, sub_file),  num_chunk_this_window_size=num_chunk_this_window_size)
    
    #load the test data from bpf_UsedForSelectWindowSize folder
    sub_test_feature_array, sub_test_label_array = data_loading_function_testset(os.path.join(SelectWindowSize_testset_dir, sub_file))
    
    sub_data_len = len(sub_label_array)
    #use the 1st half as train
    half_sub_data_len = int(sub_data_len/2)
    print('half_sub_data_len: {}'.format(half_sub_data_len), flush=True)

    sub_train_feature_array = sub_feature_array[:half_sub_data_len]
    sub_train_label_array = sub_label_array[:half_sub_data_len]

    transformed_sub_train_feature_array = featurize(sub_train_feature_array, classification_task)
    transformed_sub_test_feature_array = featurize(sub_test_feature_array, classification_task)
    
    #cross validation
    Cs = np.logspace(-5,5,11)
    
    for C in Cs:
        experiment_name = 'C{}'.format(C)
        #derived args
        result_save_subjectdir = os.path.join(result_save_rootdir, SubjectId_of_interest, experiment_name)
        result_save_subject_checkpointdir = os.path.join(result_save_subjectdir, 'checkpoint')
        result_save_subject_predictionsdir = os.path.join(result_save_subjectdir, 'predictions')
        result_save_subject_resultanalysisdir = os.path.join(result_save_subjectdir, 'result_analysis')
        result_save_subject_trainingcurvedir = os.path.join(result_save_subjectdir, 'trainingcurve')

        makedir_if_not_exist(result_save_subjectdir)
        makedir_if_not_exist(result_save_subject_checkpointdir)
        makedir_if_not_exist(result_save_subject_predictionsdir)
        makedir_if_not_exist(result_save_subject_resultanalysisdir)
        makedir_if_not_exist(result_save_subject_trainingcurvedir)
    
        result_save_dict = dict()

        if classification_task == 'binary':
            if window_size == 200:
                total_number_train_chunks = 304
                total_index = np.arange(total_number_train_chunks)
#                 train_index = total_index[:228]
#                 val_index = total_index[244:]
                train_index = total_index[:152]
                val_index = total_index[152:]
        
            elif window_size == 150:
                total_number_train_chunks = 368
                total_index = np.arange(total_number_train_chunks)
#                 train_index = total_index[:276]
#                 val_index = total_index[295:]
                train_index = total_index[:184]
                val_index = total_index[184:]
        
            elif window_size == 100:
                total_number_train_chunks = 436
                total_index = np.arange(total_number_train_chunks)
#                 train_index = total_index[:327]
#                 val_index = total_index[349:]
                train_index = total_index[:218]
                val_index = total_index[218:]
                
            elif window_size == 50:
                total_number_train_chunks = 504
                total_index = np.arange(total_number_train_chunks)
#                 train_index = total_index[:388]
#                 val_index = total_index[404:]
                train_index = total_index[:252]
                val_index = total_index[252:]
                
            elif window_size == 25:
                total_number_train_chunks = 536
                total_index = np.arange(total_number_train_chunks)
#                 train_index = total_index[:422]
#                 val_index = total_index[429:]
                train_index = total_index[:268]
                val_index = total_index[268:]
        
            elif window_size == 10:
                total_number_train_chunks = 556
                total_index = np.arange(total_number_train_chunks)
#                 train_index = total_index[:443]
#                 val_index = total_index[445:]
                train_index = total_index[:278]
                val_index = total_index[278:]
        
            else:
                raise NameError('not supported window size') 
        else:
            raise NameError('not implemented classification task')
            
        #only do 1 fold cross validation:
        #dataset object
        sub_cv_train_feature_array = transformed_sub_train_feature_array[train_index]
        sub_cv_train_label_array = sub_train_label_array[train_index]

        sub_cv_val_feature_array = transformed_sub_train_feature_array[val_index]
        sub_cv_val_label_array = sub_train_label_array[val_index]

        #create Logistic Regression object
        model = LogisticRegression(C=C, random_state=0, max_iter=5000, solver='lbfgs').fit(sub_cv_train_feature_array, sub_cv_train_label_array)

        # val performance 
        val_accuracy = model.score(sub_cv_val_feature_array, sub_cv_val_label_array) * 100
        result_save_dict['bestepoch_val_accuracy'] = val_accuracy

        # test performance
        test_accuracy = model.score(transformed_sub_test_feature_array, sub_test_label_array) * 100
        test_logits = model.predict_proba(transformed_sub_test_feature_array)
        test_class_predictions = test_logits.argmax(1)

        result_save_dict['bestepoch_test_accuracy'] = test_accuracy
        result_save_dict['bestepoch_test_logits'] = test_logits.copy()
        result_save_dict['bestepoch_test_class_labels'] = sub_test_label_array.copy()


        plot_confusion_matrix(test_class_predictions, sub_test_label_array, confusion_matrix_figure_labels, result_save_subject_resultanalysisdir, 'test_confusion_matrix.png')

        save_pickle(result_save_subject_predictionsdir, 'result_save_dict.pkl', result_save_dict)
        
        #write performance to txt file
        write_performance_info_FixedTrainValSplit('NA', result_save_subject_resultanalysisdir, val_accuracy, test_accuracy)
        
    
    
if __name__=='__main__':
    
    #parse args
    args = parser.parse_args()
    
    seed = args.seed
    data_dir = args.data_dir
    SelectWindowSize_testset_dir = args.SelectWindowSize_testset_dir
    window_size = args.window_size
    result_save_rootdir = args.result_save_rootdir
    SubjectId_of_interest = args.SubjectId_of_interest
    classification_task = args.classification_task
    
    #sanity check 
    print('type(data_dir): {}'.format(type(data_dir)))
    print('type(SelectWindowSize_testset_dir): {}'.format(type(SelectWindowSize_testset_dir)))
    print('type(window_size): {}'.format(type(window_size)))
    print('type(SubjectId_of_interest): {}'.format(type(SubjectId_of_interest)))
    print('type(result_save_rootdir): {}'.format(type(result_save_rootdir)))
    print('type(classification_task): {}'.format(type(classification_task)))
    
    args_dict = edict()
    args_dict.data_dir = data_dir
    args_dict.SelectWindowSize_testset_dir = SelectWindowSize_testset_dir
    args_dict.window_size = window_size
    args_dict.result_save_rootdir = result_save_rootdir
    args_dict.SubjectId_of_interest = SubjectId_of_interest
    args_dict.classification_task = classification_task
    
    seed_everything(seed)
    train_classifier(args_dict)
        
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    

