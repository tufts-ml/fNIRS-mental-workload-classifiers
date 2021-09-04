import os
import sys
import numpy as np
import argparse

from easydict import EasyDict as edict
from tqdm import trange
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier as rfc

sys.path.insert(0, '/cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental-workload-classifiers/helpers/')
import models
import brain_data
from utils import seed_everything, featurize, makedir_if_not_exist, plot_confusion_matrix, save_pickle, write_performance_info_FixedTrainValSplit

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--data_dir', default='../data/Leon/Visual/size_2sec_10ts_stride_3ts/', help='folder to the train data')
parser.add_argument('--window_size', default=200, type=int, help='window size')
parser.add_argument('--result_save_rootdir', default='./experiments', help='folder to the result')
parser.add_argument('--classification_task', default='four_class', help='binary or four-class classification')
parser.add_argument('--setting', default='seed1', help='which predefined train val test split scenario')

def train_classifier(args_dict, train_subjects, val_subjects, test_subjects_URG, test_subjects_WHITE, test_subjects_ASIAN):
    #convert to string list
    train_subjects = [str(i) for i in train_subjects]
    val_subjects = [str(i) for i in val_subjects]
    test_subjects_URG = [str(i) for i in test_subjects_URG]
    test_subjects_WHITE = [str(i) for i in test_subjects_WHITE]
    test_subjects_ASIAN = [str(i) for i in test_subjects_ASIAN]
    
    #combined the test subjects
    test_subjects = test_subjects_URG + test_subjects_WHITE + test_subjects_ASIAN

    #parse args:
    data_dir = args_dict.data_dir
    window_size = args_dict.window_size
    result_save_rootdir = args_dict.result_save_rootdir
    classification_task = args_dict.classification_task
    
    num_chunk_this_window_size = 1488
        
        
    if classification_task == 'binary':
        data_loading_function = brain_data.read_subject_csv_binary
        confusion_matrix_figure_labels = ['0back', '2back']
        
#     elif classification_task == 'four_class':
#         data_loading_function = brain_data.read_subject_csv
#         confusion_matrix_figure_labels = ['0back', '1back', '2back', '3back']
        
    else:
        raise NameError('not supported classification type')
        
    
    #create the group train data 
    group_model_sub_train_feature_list = []
    group_model_sub_train_label_list = []
    
    for subject in train_subjects:
        sub_feature, sub_label = data_loading_function(os.path.join(data_dir, 'sub_{}.csv'.format(subject)), num_chunk_this_window_size=num_chunk_this_window_size)
        
        group_model_sub_train_feature_list.append(sub_feature)
        group_model_sub_train_label_list.append(sub_label)
    
    group_model_sub_train_feature_array = np.concatenate(group_model_sub_train_feature_list, axis=0).astype(np.float32)
    group_model_sub_train_label_array = np.concatenate(group_model_sub_train_label_list, axis=0)
    
    transformed_group_model_sub_train_feature_array = featurize(group_model_sub_train_feature_array, classification_task)
    
    
    #create the group val data
    group_model_sub_val_feature_list = []
    group_model_sub_val_label_list = []
    
    for subject in val_subjects:
        sub_feature, sub_label = data_loading_function(os.path.join(data_dir, 'sub_{}.csv'.format(subject)), num_chunk_this_window_size=num_chunk_this_window_size)
        
        group_model_sub_val_feature_list.append(sub_feature)
        group_model_sub_val_label_list.append(sub_label)
    
    group_model_sub_val_feature_array = np.concatenate(group_model_sub_val_feature_list, axis=0).astype(np.float32)
    group_model_sub_val_label_array = np.concatenate(group_model_sub_val_label_list, axis=0)
    
    transformed_group_model_sub_val_feature_array = featurize(group_model_sub_val_feature_array, classification_task)

    
    #cross validation
    max_features_list = [0.166, 0.333, 0.667, 0.1]
    min_samples_leaf_list = [4, 16, 64]   
    
    for max_features in max_features_list:
        for min_samples_leaf in min_samples_leaf_list:
            experiment_name = 'MaxFeatures{}_MinSamplesLeaf{}'.format(max_features, min_samples_leaf)
            
            #create test subjects dict
            test_subjects_dict = dict()
            for test_subject in test_subjects:
                #load this subject's test data
                sub_feature_array, sub_label_array = data_loading_function(os.path.join(data_dir, 'sub_{}.csv'.format(test_subject)), num_chunk_this_window_size=num_chunk_this_window_size)
                
                sub_data_len = len(sub_label_array)
                assert sub_data_len == int(num_chunk_this_window_size/2), 'subject {} len is not {} for binary classification'.format(test_subject, int(num_chunk_this_window_size/2))
                half_sub_data_len = int(sub_data_len/2)
                print('half_sub_data_len: {}'.format(half_sub_data_len), flush=True)

                sub_test_feature_array = sub_feature_array[half_sub_data_len:]
                transformed_sub_test_feature_array = featurize(sub_test_feature_array, classification_task)
                sub_test_label_array = sub_label_array[half_sub_data_len:]

                #create the dict for this subject: 
                #each subject's dict has: 'transformed_sub_test_feature_array', 'sub_test_label_array',
                                    # 'resutl_save_subjectdir', 'resutl_save_subject_checkpointdir', 
                                    # 'result_save_subject_predictiondir', 'result_save_subject_resultanalysisdir'
                                    # 'result_save_subject_trainingcurvedir', 'result_save_dir', 

                test_subjects_dict[test_subject] = dict()
                test_subjects_dict[test_subject]['transformed_sub_test_feature_array'] = transformed_sub_test_feature_array
                test_subjects_dict[test_subject]['sub_test_label_array'] = sub_test_label_array

            
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
                
                test_subjects_dict[test_subject]['result_save_subjectdir'] = result_save_subjectdir
                test_subjects_dict[test_subject]['result_save_subject_checkpointdir'] = result_save_subject_checkpointdir
                test_subjects_dict[test_subject]['result_save_subject_predictionsdir'] = result_save_subject_predictionsdir
                test_subjects_dict[test_subject]['result_save_subject_resultanalysisdir'] = result_save_subject_resultanalysisdir
                test_subjects_dict[test_subject]['result_save_subject_trainingcurvedir'] = result_save_subject_trainingcurvedir

                test_subjects_dict[test_subject]['result_save_dict'] = dict()
            
            
            #create Logistic Regression object
            model = rfc(max_features=max_features, min_samples_leaf=min_samples_leaf).fit(transformed_group_model_sub_train_feature_array, group_model_sub_train_label_array)
            
            # val performance 
            val_accuracy = model.score(transformed_group_model_sub_val_feature_array, group_model_sub_val_label_array) * 100
            
            # test performance
            for test_subject in test_subjects:
                test_subjects_dict[test_subject]['result_save_dict']['bestepoch_val_accuracy'] = val_accuracy
                test_accuracy = model.score(test_subjects_dict[test_subject]['transformed_sub_test_feature_array'], test_subjects_dict[test_subject]['sub_test_label_array']) * 100
                test_logits = model.predict_proba(test_subjects_dict[test_subject]['transformed_sub_test_feature_array'])
                test_class_predictions = test_logits.argmax(1)
            
                test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_accuracy'] = test_accuracy
                test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_logits'] = test_logits
                test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_class_labels'] = test_subjects_dict[test_subject]['sub_test_label_array']
                
                plot_confusion_matrix(test_class_predictions, test_subjects_dict[test_subject]['sub_test_label_array'], confusion_matrix_figure_labels, test_subjects_dict[test_subject]['result_save_subject_resultanalysisdir'], 'test_confusion_matrix.png')

                save_pickle(test_subjects_dict[test_subject]['result_save_subject_predictionsdir'], 'result_save_dict.pkl', test_subjects_dict[test_subject]['result_save_dict'])
                
                #write performance to txt file
                write_performance_info_FixedTrainValSplit('NA', test_subjects_dict[test_subject]['result_save_subject_resultanalysisdir'], val_accuracy, test_accuracy)

    
    
if __name__=='__main__':
    
    #parse args
    args = parser.parse_args()
    
    seed = args.seed
    data_dir = args.data_dir
    window_size = args.window_size
    result_save_rootdir = args.result_save_rootdir
    classification_task = args.classification_task
    setting = args.setting
        
    if setting == 'random_partition1':
        train_subjects = [72, 68, 27, 75, 52, 46, 65, 71, 73, 63, 74, 44, 81, 13, 54, 84, 55, 37, 51, 7]
        val_subjects = [56, 57, 24, 43, 25, 61]
        test_subjects_URG = [22, 70, 78, 28, 60, 58]
        test_subjects_WHITE = [85, 38, 42, 29, 48, 40]
        test_subjects_ASIAN = [5, 76, 93, 49, 94, 35]
    
    elif setting == 'random_partition2':
        train_subjects = [27, 84, 7, 49, 68, 37, 76, 71, 72, 56, 5, 93, 55, 52, 94, 46, 61, 81, 74, 43]
        val_subjects = [65, 25, 51, 13, 63, 75]
        test_subjects_URG = [22, 70, 78, 28, 60, 58]
        test_subjects_WHITE = [80, 15, 82, 29, 32, 92]
        test_subjects_ASIAN = [54, 44, 35, 73, 24, 57]

    elif setting == 'random_partition3':
        train_subjects = [55, 25, 71, 24, 68, 49, 46, 13, 44, 5, 65, 54, 84, 61, 63, 7, 27, 74, 43, 73]
        val_subjects = [72, 76, 51, 57, 75, 81]
        test_subjects_URG = [22, 70, 78, 28, 60, 58]
        test_subjects_WHITE = [79, 95, 31, 32, 92, 34]
        test_subjects_ASIAN = [94, 56, 93, 35, 37, 52]
    
    elif setting == 'random_partition4':
        train_subjects = [25, 63, 56, 74, 72, 76, 7, 24, 46, 43, 5, 13, 71, 44, 51, 52, 27, 94, 55, 81]
        val_subjects = [57, 37, 68, 75, 54, 65]
        test_subjects_URG = [22, 70, 78, 28, 60, 58]
        test_subjects_WHITE = [42, 32, 14, 95, 92, 97]
        test_subjects_ASIAN = [61, 84, 73, 93, 49, 35]
    
    else:
        raise NameError('not supported setting')
    
    
    
    #sanity check 
    print('data_dir: {}, type: {}'.format(data_dir, type(data_dir)))
    print('window_size: {}, type: {}'.format(window_size, type(window_size)))
    print('result_save_rootdir: {}, type: {}'.format(result_save_rootdir, type(result_save_rootdir)))
    print('classification_task: {}, type: {}'.format(classification_task, type(classification_task)))
    print('setting: {} type: {}'.format(setting, type(setting)))
    
    args_dict = edict()
    args_dict.data_dir = data_dir
    args_dict.window_size = window_size
    args_dict.result_save_rootdir = result_save_rootdir
    args_dict.classification_task = classification_task
    
    seed_everything(seed)
    train_classifier(args_dict, train_subjects, val_subjects, test_subjects_URG, test_subjects_WHITE, test_subjects_ASIAN)
        
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    

