import os
import sys
import numpy as np
import argparse

from easydict import EasyDict as edict
from tqdm import trange
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier as rfc

sys.path.insert(0, '/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/helpers/')
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

def train_classifier(args_dict, train_subjects, val_subjects, test_subjects_FEMALE, test_subjects_MALE):
    #convert to string list
    train_subjects = [str(i) for i in train_subjects]
    val_subjects = [str(i) for i in val_subjects]
    test_subjects_FEMALE = [str(i) for i in test_subjects_FEMALE]
    test_subjects_MALE = [str(i) for i in test_subjects_MALE]
    
    #combined the test subjects
    test_subjects = test_subjects_FEMALE + test_subjects_MALE
    
    #parse args:
    data_dir = args_dict.data_dir
    window_size = args_dict.window_size
    result_save_rootdir = args_dict.result_save_rootdir
    classification_task = args_dict.classification_task
    
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
        confusion_matrix_figure_labels = ['0back', '1back', '2back', '3back']
        
    elif classification_task == 'binary':
        data_loading_function = brain_data.read_subject_csv_binary
        confusion_matrix_figure_labels = ['0back', '2back']
        
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
    
    
    if setting == 'seed1':
        train_subjects = [28, 21, 32, 61, 37, 46, 94, 55, 85, 40, 25, 24, 91, 22, 15, 23, 60, 20, 76, 45, 82, 27, 65, 73, 97]
        val_subjects = [84, 95, 38, 93, 7, 51, 62, 5]
        test_subjects_FEMALE = [13, 79, 14, 57, 71, 74]
        test_subjects_MALE = [78, 68, 72, 31, 83, 56]
        
    elif setting == 'seed2':
        train_subjects = [93, 60, 91, 32, 94, 97, 71, 28, 5, 37, 15, 82, 74, 95, 57, 23, 62, 7, 46, 27, 20, 45, 79, 24, 21]
        val_subjects = [65, 76, 25, 38, 84, 55, 51, 40]
        test_subjects_FEMALE = [22, 61, 85, 13, 14, 73]
        test_subjects_MALE = [78, 43, 31, 72, 58, 81]
    
    elif setting == 'seed3':
        train_subjects = [93, 73, 51, 5, 94, 38, 62, 76, 27, 55, 60, 95, 97, 85, 79, 61, 20, 45, 23, 57, 7, 40, 91, 15, 82]
        val_subjects = [13, 32, 74, 22, 84, 46, 37, 71]
        test_subjects_FEMALE = [25, 28, 24, 14, 21, 65]
        test_subjects_MALE = [56, 68, 29, 48, 54, 58]

    elif setting == 'seed4':
        train_subjects = [91, 85, 73, 93, 74, 79, 95, 45, 7, 37, 28, 27, 97, 61, 20, 23, 22, 32, 84, 24, 46, 25, 65, 13, 5]
        val_subjects = [55, 40, 76, 15, 51, 14, 82, 94]
        test_subjects_FEMALE = [21, 60, 38, 71, 57, 62]
        test_subjects_MALE = [83, 68, 86, 81, 75, 54]
    
    elif setting == 'seed5':
        train_subjects = [62, 73, 15, 93, 21, 27, 74, 40, 14, 25, 23, 20, 55, 95, 7, 85, 13, 97, 57, 61, 94, 37, 65, 84, 24]
        val_subjects = [51, 79, 38, 76, 5, 60, 28, 45]
        test_subjects_FEMALE = [82, 46, 32, 71, 22, 91]
        test_subjects_MALE = [83, 86, 54, 92, 44, 80]

    elif setting == 'seed6':
        train_subjects = [37, 32, 21, 51, 7, 93, 61, 25, 76, 15, 85, 46, 14, 45, 91, 94, 22, 84, 40, 62, 79, 20, 65, 23, 5]
        val_subjects = [74, 57, 13, 27, 95, 38, 28, 97]
        test_subjects_FEMALE = [55, 60, 73, 71, 24, 82]
        test_subjects_MALE = [44, 42, 31, 83, 78, 64]
    
    elif setting == 'seed7':
        train_subjects = [79, 21, 82, 40, 13, 85, 95, 28, 76, 7, 61, 57, 97, 55, 15, 74, 93, 24, 32, 27, 25, 37, 38, 23, 45]
        val_subjects = [84, 46, 71, 5, 14, 62, 65, 91]
        test_subjects_FEMALE = [94, 60, 22, 73, 51, 20]
        test_subjects_MALE = [31, 36, 92, 1, 86, 81]
    
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
    train_classifier(args_dict, train_subjects, val_subjects, test_subjects_FEMALE, test_subjects_MALE)
        
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    

