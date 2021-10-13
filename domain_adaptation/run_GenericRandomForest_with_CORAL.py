import os
import sys
import numpy as np
import argparse

import time

from easydict import EasyDict as edict
from tqdm import trange
# from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier as rfc

#for CORAL
import scipy.io
import scipy.linalg

YOUR_PATH = os.environ['YOUR_PATH']
sys.path.insert(0, os.path.join(YOUR_PATH, 'fNIRS-mental-workload-classifiers/helpers'))
import models
import brain_data
from utils import generic_GetTrainValTestSubjects, seed_everything, featurize, makedir_if_not_exist, plot_confusion_matrix, save_pickle, write_performance_info_FixedTrainValSplit, write_program_time, write_inference_time

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--data_dir', default='../data/Leon/Visual/size_2sec_10ts_stride_3ts/', help='folder to the train data')
parser.add_argument('--window_size', default=10, type=int, help='window size')
parser.add_argument('--classification_task', default='four_class', help='binary or four-class classification')
parser.add_argument('--result_save_rootdir', default='./experiments', help='folder to the result')
parser.add_argument('--setting', default='train64test7_bucket1', help='which predefined train test split scenario')

#parameter for CORAL domain adapation
parser.add_argument('--adapt_on', default='train_100', help="what portion of the test subject' train set is used for adaptation")

#CORAL implementation:
#https://github.com/jindongwang/transferlearning/blob/master/code/traditional/CORAL/CORAL.py
def CoralTransform(Xs, Xt):
    '''
    Perform CORAL on the source domain features
    :param Xs: ns * n_feature, source feature
    :param Xt: nt * n_feature, target feature
    :return: New source domain features
    '''
    cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
    cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
    
    A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                    scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
    
    Xs_new = np.real(np.dot(Xs, A_coral))
    return Xs_new
    
    

def train_classifier(args_dict, train_subjects, val_subjects, test_subjects):
    
    #convert to string list
    train_subjects = [str(i) for i in train_subjects]
    val_subjects = [str(i) for i in val_subjects]
    test_subjects = [str(i) for i in test_subjects]
        
    #parse args:
    data_dir = args_dict.data_dir
    window_size = args_dict.window_size
    classification_task = args_dict.classification_task    
    result_save_rootdir = args_dict.result_save_rootdir
#     setting = args_dict.setting  #does not need 'setting' inside train_classifier  
    adapt_on = args_dict.adapt_on
    num_chunk_this_window_size = 1488

    
    if classification_task == 'binary':
        data_loading_function = brain_data.read_subject_csv_binary
        confusion_matrix_figure_labels = ['0back', '2back']
        
#     elif classification_task == 'four_class':
#         data_loading_function = brain_data.read_subject_csv
#         confusion_matrix_figure_labels = ['0back', '1back', '2back', '3back']
        
    else:
        raise NameError('not supported classification type')
        
    
        
    #create the group data
    group_model_sub_train_feature_list = []
    group_model_sub_train_label_list = []
    
    for subject in train_subjects:
        sub_feature, sub_label = data_loading_function(os.path.join(data_dir, 'sub_{}.csv'.format(subject)),  num_chunk_this_window_size=num_chunk_this_window_size)
        
        group_model_sub_train_feature_list.append(sub_feature)
        group_model_sub_train_label_list.append(sub_label)
    
    group_model_sub_train_feature_array = np.concatenate(group_model_sub_train_feature_list, axis=0).astype(np.float32)
    group_model_sub_train_label_array = np.concatenate(group_model_sub_train_label_list, axis=0)
    
    transformed_group_model_sub_train_feature_array = featurize(group_model_sub_train_feature_array, classification_task)
    
    
    
    #create the group val data
    group_model_sub_val_feature_list = []
    group_model_sub_val_label_list = []
    
    for subject in val_subjects:
        sub_feature, sub_label = data_loading_function(os.path.join(data_dir, 'sub_{}.csv'.format(subject)),  num_chunk_this_window_size=num_chunk_this_window_size)
        
        group_model_sub_val_feature_list.append(sub_feature)
        group_model_sub_val_label_list.append(sub_label)
    
    group_model_sub_val_feature_array = np.concatenate(group_model_sub_val_feature_list, axis=0).astype(np.float32)
    group_model_sub_val_label_array = np.concatenate(group_model_sub_val_label_list, axis=0)
    
    transformed_group_model_sub_val_feature_array = featurize(group_model_sub_val_feature_array, classification_task)

    
    
    #Perform domain adapation for each test subject in this bucket
    for test_subject in test_subjects:
        
        #load this subject's test data
        sub_feature_array, sub_label_array = data_loading_function(os.path.join(data_dir, 'sub_{}.csv'.format(test_subject)), num_chunk_this_window_size=num_chunk_this_window_size)
        
        #sainty check for this test subject's data
        sub_data_len = len(sub_label_array)
        assert sub_data_len == int(num_chunk_this_window_size/2), 'subject {} len is not {} for binary classification'.format(test_subject, int(num_chunk_this_window_size/2))
        
        half_sub_data_len = int(sub_data_len/2)
        print('half_sub_data_len: {}'.format(half_sub_data_len), flush=True)
        
        #first half of the test subject's data is train set, the second half is test set
        sub_test_feature_array = sub_feature_array[half_sub_data_len:]
        
        transformed_sub_test_feature_array = featurize(sub_test_feature_array, classification_task)
        sub_test_label_array = sub_label_array[half_sub_data_len:]

        
        sub_adapt_feature_array = sub_feature_array[:half_sub_data_len]
        if adapt_on == 'train_100':
            transformed_sub_adapt_feature_array = featurize(sub_adapt_feature_array, classification_task)
            print('adapt on data size: {}'.format(len(transformed_sub_adapt_feature_array)))
            
        elif adapt_on == 'train_50':
            transformed_sub_adapt_feature_array = featurize(sub_adapt_feature_array[-int(0.5*half_sub_data_len):], classification_task)
            print('adapt on data size: {}'.format(len(transformed_sub_adapt_feature_array)))
        
        else:
            raise NameError('on the predefined gride')

            
        start_time = time.time()
        CORAL_group_model_sub_train_feature_array = CoralTransform(transformed_group_model_sub_train_feature_array, transformed_sub_adapt_feature_array)
        CORAL_group_model_sub_val_feature_array = CoralTransform(transformed_group_model_sub_val_feature_array, transformed_sub_adapt_feature_array)
        
        
        #cross validation
        max_features_list = [0.166, 0.333, 0.667, 0.1]
        min_samples_leaf_list = [4, 16, 64]  
        

        for max_features in max_features_list:
            for min_samples_leaf in min_samples_leaf_list:
                experiment_name = 'MaxFeatures{}_MinSamplesLeaf{}'.format(max_features, min_samples_leaf)

       
                #derived args
                result_save_subjectdir = os.path.join(result_save_rootdir, test_subject, experiment_name)
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

                #create Logistic Regression object
                model =rfc(max_features=max_features, min_samples_leaf=min_samples_leaf).fit(CORAL_group_model_sub_train_feature_array, group_model_sub_train_label_array)

                # val performance 
                val_accuracy = model.score(CORAL_group_model_sub_val_feature_array, group_model_sub_val_label_array) * 100

                result_save_dict['bestepoch_val_accuracy'] = val_accuracy

                # test performance
                inference_start_time = time.time()
                test_accuracy = model.score(transformed_sub_test_feature_array, sub_test_label_array) * 100
                test_logits = model.predict_proba(transformed_sub_test_feature_array)
                test_class_predictions = test_logits.argmax(1)
                inference_end_time = time.time()
                inference_time = inference_end_time - inference_start_time

                result_save_dict['bestepoch_test_accuracy'] = test_accuracy
                result_save_dict['bestepoch_test_logits'] = test_logits.copy()
                result_save_dict['bestepoch_test_class_labels'] = sub_test_label_array.copy()

                plot_confusion_matrix(test_class_predictions, sub_test_label_array, confusion_matrix_figure_labels, result_save_subject_resultanalysisdir, 'test_confusion_matrix.png')

                save_pickle(result_save_subject_predictionsdir, 'result_save_dict.pkl', result_save_dict)

                #write performance to txt file
                write_performance_info_FixedTrainValSplit('NA', result_save_subject_resultanalysisdir, val_accuracy, test_accuracy)
        
        end_time = time.time()
        total_time = end_time - start_time
        write_program_time(result_save_rootdir, total_time)
        write_inference_time(result_save_rootdir, inference_time)



    
if __name__=='__main__':
    
    #parse args
    args = parser.parse_args()
    
    seed = args.seed
    data_dir = args.data_dir
    window_size = args.window_size
    classification_task = args.classification_task
    result_save_rootdir = args.result_save_rootdir
    setting = args.setting
    adapt_on = args.adapt_on
    
    test_subjects, train_subjects, val_subjects = generic_GetTrainValTestSubjects(setting)
    
    #sanity check 
    print('data_dir: {} type: {}'.format(data_dir, type(data_dir)))
    print('window_size: {} type: {}'.format(window_size, type(window_size)))
    print('classification_task: {} type: {}'.format(classification_task, type(classification_task)))
    print('result_save_rootdir: {} type: {}'.format(result_save_rootdir, type(result_save_rootdir)))
    print('setting: {} type: {}'.format(setting, type(setting)))
    print('adapt_on: {} type: {}'.format(adapt_on, type(adapt_on)))

    
    args_dict = edict()
    args_dict.data_dir = data_dir
    args_dict.window_size = window_size
    args_dict.classification_task = classification_task
    args_dict.result_save_rootdir = result_save_rootdir
#     args_dict.setting = setting #does not need 'setting' inside train_classifier 
    args_dict.adapt_on = adapt_on
    
    seed_everything(seed)
    train_classifier(args_dict, train_subjects, val_subjects, test_subjects)
        
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    

