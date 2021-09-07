import os
import sys
import numpy as np
import torch
import torch.nn as nn

import argparse
import time

from easydict import EasyDict as edict
from tqdm import trange

sys.path.insert(0, '/cluster/tufts/hugheslab/zhuang12/HCI/fNIRS-mental-workload-classifiers/helpers/')
import models
import brain_data
from utils import seed_everything, makedir_if_not_exist, plot_confusion_matrix, save_pickle, train_one_epoch, eval_model, save_training_curves_FixedTrainValSplit, write_performance_info_FixedTrainValSplit, write_program_time, write_inference_time

# from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int, help="random seed")
parser.add_argument('--gpu_idx', default=0, type=int, help="gpu idx")
parser.add_argument('--data_dir', default='../data/Leon/Visual/size_40sec_200ts_stride_3ts/', help="folder to the dataset")
parser.add_argument('--window_size', default=200, type=int, help='window size')
parser.add_argument('--result_save_rootdir', default='./experiments', help="Directory containing the dataset")
parser.add_argument('--classification_task', default='four_class', help='binary or four-class classification')
parser.add_argument('--restore_file', default='None', help="xxx.statedict")
parser.add_argument('--n_epoch', default=100, type=int, help="number of epoch")
parser.add_argument('--setting', default='64vs4_TestBucket1', help='which predefined train val test split scenario')


#for personal model, save the test prediction of each cv fold
def train_classifier(args_dict, train_subjects, val_subjects, test_subjects):
    
    #convert to string list
    train_subjects = [str(i) for i in train_subjects]
    val_subjects = [str(i) for i in val_subjects]
    test_subjects = [str(i) for i in test_subjects]
    
    #parse args:
    gpu_idx = args_dict.gpu_idx
    data_dir = args_dict.data_dir
    window_size = args_dict.window_size
    result_save_rootdir = args_dict.result_save_rootdir
    classification_task = args_dict.classification_task
    restore_file = args_dict.restore_file
    n_epoch = args_dict.n_epoch
    
    model_to_use = models.DeepConvNet150
    num_chunk_this_window_size = 1488
    

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
        sub_feature, sub_label = data_loading_function(os.path.join(data_dir, 'sub_{}.csv'.format(subject)),  num_chunk_this_window_size=num_chunk_this_window_size)
        
        group_model_sub_train_feature_list.append(sub_feature)
        group_model_sub_train_label_list.append(sub_label)
    
    group_model_sub_train_feature_array = np.concatenate(group_model_sub_train_feature_list, axis=0).astype(np.float32)
    group_model_sub_train_label_array = np.concatenate(group_model_sub_train_label_list, axis=0)
    
    
    #create the group val data
    group_model_sub_val_feature_list = []
    group_model_sub_val_label_list = []
    
    for subject in val_subjects:
        sub_feature, sub_label = data_loading_function(os.path.join(data_dir, 'sub_{}.csv'.format(subject)),  num_chunk_this_window_size=num_chunk_this_window_size)
        
        group_model_sub_val_feature_list.append(sub_feature)
        group_model_sub_val_label_list.append(sub_label)
    
    group_model_sub_val_feature_array = np.concatenate(group_model_sub_val_feature_list, axis=0).astype(np.float32)
    group_model_sub_val_label_array = np.concatenate(group_model_sub_val_label_list, axis=0)
    
    
    #dataset object
    group_train_set = brain_data.brain_dataset(group_model_sub_train_feature_array, group_model_sub_train_label_array)
    group_val_set = brain_data.brain_dataset(group_model_sub_val_feature_array, group_model_sub_val_label_array)

    #dataloader object
    cv_train_batch_size = len(group_train_set)
    cv_val_batch_size = len(group_val_set)
    group_train_loader = torch.utils.data.DataLoader(group_train_set, batch_size=cv_train_batch_size, shuffle=True) 
    group_val_loader = torch.utils.data.DataLoader(group_val_set, batch_size=cv_val_batch_size, shuffle=False)
  
    #GPU setting
    cuda = torch.cuda.is_available()
    if cuda:
        print('Detected GPUs', flush = True)
        device = torch.device('cuda')
#         device = torch.device('cuda:{}'.format(gpu_idx))
    else:
        print('DID NOT detect GPUs', flush = True)
        device = torch.device('cpu')
        

    #cross validation:
    lrs = [0.001, 0.01, 0.1, 1.0, 10.0]
    dropouts = [0.25, 0.5, 0.75]
    
    start_time = time.time()
    
    for lr in lrs:
        for dropout in dropouts:
            experiment_name = 'lr{}_dropout{}'.format(lr, dropout)#experiment name: used for indicating hyper setting

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
                sub_test_label_array = sub_label_array[half_sub_data_len:]
               
                #convert subject's test data into dataset object
                sub_test_set = brain_data.brain_dataset(sub_test_feature_array, sub_test_label_array)

                #convert subject's test dataset object into dataloader object
                test_batch_size = len(sub_test_set)
                sub_test_loader = torch.utils.data.DataLoader(sub_test_set, batch_size=test_batch_size, shuffle=False)
                
                #create the dict for this subject: 
                #each subject's dict has: 'transformed_sub_test_feature_array', 'sub_test_label_array',
                                    # 'resutl_save_subjectdir', 'resutl_save_subject_checkpointdir', 
                                    # 'result_save_subject_predictiondir', 'result_save_subject_resultanalysisdir'
                                    # 'result_save_subject_trainingcurvedir', 'result_save_dir', 
            
                test_subjects_dict[test_subject] = dict()

                test_subjects_dict[test_subject]['sub_test_loader'] = sub_test_loader
                test_subjects_dict[test_subject]['sub_test_label_array'] = sub_test_label_array
                

                #derived arg
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
                

            #create model
            model = model_to_use(dropout=dropout).to(device)

            #reload weights from restore_file is specified
            if restore_file != 'None':
                restore_path = os.path.join(os.path.join(result_save_subject_checkpointdir, restore_file))
                print('loading checkpoint: {}'.format(restore_path))
                model.load_state_dict(torch.load(restore_path, map_location=device))

            #create criterion and optimizer
            criterion = nn.NLLLoss() #for EEGNet and DeepConvNet, use nn.NLLLoss directly, which accept integer labels
            optimizer = torch.optim.Adam(model.parameters(), lr=lr) #the authors used Adam instead of SGD
#             optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
            #training loop
            best_val_accuracy = 0.0

            epoch_train_loss = []
            epoch_train_accuracy = []
            epoch_validation_accuracy = []

            for epoch in trange(n_epoch, desc='1-fold cross validation'):
                average_loss_this_epoch = train_one_epoch(model, optimizer, criterion, group_train_loader, device)
                val_accuracy, _, _, _ = eval_model(model, group_val_loader, device)
                train_accuracy, _, _ , _ = eval_model(model, group_train_loader, device)

                epoch_train_loss.append(average_loss_this_epoch)
                epoch_train_accuracy.append(train_accuracy)
                epoch_validation_accuracy.append(val_accuracy)

                #update is_best flag
                is_best = val_accuracy >= best_val_accuracy
                
                if is_best:
                    best_val_accuracy = val_accuracy
                    
                    for test_subject in test_subjects:
                        torch.save(model.state_dict(), os.path.join(test_subjects_dict[test_subject]['result_save_subject_checkpointdir'], 'best_model.statedict'))
                                                
                        inference_start_time = time.time()
                        test_accuracy, test_class_predictions, test_class_labels, test_logits = eval_model(model, test_subjects_dict[test_subject]['sub_test_loader'], device)
                        inference_end_time = time.time()
                        infernece_time = inference_end_time - inference_start_time
                        
                        print('test accuracy this epoch for subject: {} is {}'.format(test_subject, test_accuracy))
                        test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_accuracy'] = test_accuracy
                        test_subjects_dict[test_subject]['result_save_dict']['bestepoch_val_accuracy'] = val_accuracy
                        test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_logits'] = test_logits.copy()
                        test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_class_predictions'] = test_class_predictions.copy()
                        test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_class_labels'] = test_class_labels.copy()


            for test_subject in test_subjects:
                
                #save training curve for each fold
                save_training_curves_FixedTrainValSplit('training_curve.png', test_subjects_dict[test_subject]['result_save_subject_trainingcurvedir'], epoch_train_loss, epoch_train_accuracy, epoch_validation_accuracy)
                
                #confusion matrix 
                plot_confusion_matrix(test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_class_predictions'], test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_class_labels'], confusion_matrix_figure_labels, test_subjects_dict[test_subject]['result_save_subject_resultanalysisdir'], 'test_confusion_matrix.png')

                #save the model at last epoch
                torch.save(model.state_dict(), os.path.join(test_subjects_dict[test_subject]['result_save_subject_checkpointdir'], 'last_model.statedict'))


                #save result_save_dict
                save_pickle(test_subjects_dict[test_subject]['result_save_subject_predictionsdir'], 'result_save_dict.pkl', test_subjects_dict[test_subject]['result_save_dict'])

                #write performance to txt file
                write_performance_info_FixedTrainValSplit(model.state_dict(), test_subjects_dict[test_subject]['result_save_subject_resultanalysisdir'], test_subjects_dict[test_subject]['result_save_dict']['bestepoch_val_accuracy'], test_subjects_dict[test_subject]['result_save_dict']['bestepoch_test_accuracy'])

    end_time = time.time()
    total_time = end_time - start_time
    write_program_time(result_save_rootdir, total_time)
    write_inference_time(result_save_rootdir, inference_time)


if __name__=='__main__':
    
    #parse args
    args = parser.parse_args()
    
    seed = args.seed
    gpu_idx = args.gpu_idx
    data_dir = args.data_dir
    window_size = args.window_size
    result_save_rootdir = args.result_save_rootdir
    classification_task = args.classification_task
    restore_file = args.restore_file
    n_epoch = args.n_epoch
    setting = args.setting

    if setting == '64vs4_TestBucket1':
        test_subjects = [86, 56, 72, 79]
        train_subjects = [5, 40, 35, 14, 65, 49, 32, 42, 25, 15, 81, 83, 38, 34, 60, 13, 78, 57, 36, 80, 27, 20, 61, 85, 23, 54, 28, 84, 31, 1, 73, 55, 22, 92, 58, 95, 93, 29, 69, 82, 97, 45, 7, 46, 91, 75, 24, 74]
        val_subjects = [37, 63, 21, 52, 43, 94, 62, 68, 70, 64, 71, 51, 76, 44, 48, 47]
        
    elif setting == '64vs4_TestBucket2':
        test_subjects = [93, 82, 55, 48]
        train_subjects = [81, 60, 79, 29, 78, 36, 22, 51, 80, 97, 37, 71, 49, 47, 25, 62, 20, 74, 7, 84, 54, 42, 68, 70, 83, 5, 92, 24, 1, 85, 76, 86, 40, 64, 95, 69, 28, 27, 15, 14, 63, 13, 75, 23, 58, 38, 35, 34]
        val_subjects = [94, 52, 43, 44, 91, 65, 72, 31, 46, 57, 45, 32, 21, 61, 56, 73]
    
    elif setting == '64vs4_TestBucket3':
        test_subjects = [80, 14, 58, 75]
        train_subjects = [82, 1, 38, 95, 23, 86, 56, 71, 79, 72, 24, 35, 36, 43, 40, 74, 45, 92, 49, 15, 25, 73, 65, 47, 63, 64, 51, 32, 44, 97, 7, 29, 62, 52, 61, 68, 83, 57, 13, 94, 70, 27, 46, 31, 60, 54, 84, 85]
        val_subjects = [93, 78, 21, 55, 22, 48, 34, 5, 91, 81, 76, 28, 42, 20, 69, 37]
        
    elif setting == '64vs4_TestBucket4':
        test_subjects = [62, 47, 52, 84]
        train_subjects = [78, 51, 27, 49, 82, 23, 46, 85, 74, 36, 86, 25, 7, 75, 32, 79, 31, 22, 92, 28, 34, 71, 20, 65, 56, 73, 57, 1, 81, 83, 24, 58, 69, 95, 38, 91, 13, 54, 97, 42, 93, 80, 15, 35, 43, 21, 55, 72]
        val_subjects = [5, 60, 76, 40, 63, 14, 44, 37, 68, 61, 64, 29, 45, 48, 70, 94]
    
    elif setting == '64vs4_TestBucket5':
        test_subjects = [73, 69, 42, 63]
        train_subjects = [72, 97, 84, 62, 54, 29, 20, 32, 60, 35, 52, 51, 15, 45, 80, 56, 70, 81, 79, 94, 91, 28, 58, 48, 34, 55, 13, 83, 40, 27, 37, 93, 61, 57, 82, 21, 24, 47, 5, 46, 31, 43, 14, 95, 68, 25, 49, 44]
        val_subjects = [76, 75, 22, 36, 7, 86, 23, 71, 38, 64, 65, 85, 78, 74, 92, 1]
    
    elif setting == '64vs4_TestBucket6':
        test_subjects = [81, 15, 57, 70]
        train_subjects = [7, 68, 38, 29, 79, 45, 83, 76, 82, 63, 48, 13, 75, 80, 28, 93, 58, 23, 60, 43, 64, 47, 14, 49, 78, 40, 52, 36, 34, 42, 20, 94, 44, 24, 31, 25, 65, 69, 74, 37, 91, 62, 71, 95, 35, 85, 92, 55]
        val_subjects = [56, 1, 54, 27, 21, 46, 61, 86, 97, 51, 32, 84, 5, 72, 73, 22]
    
    elif setting == '64vs4_TestBucket7':
        test_subjects = [27, 92, 38, 76]
        train_subjects = [49, 82, 57, 63, 60, 23, 47, 36, 48, 5, 32, 51, 58, 70, 80, 46, 69, 34, 21, 28, 43, 83, 7, 97, 45, 13, 86, 72, 54, 24, 94, 25, 14, 40, 65, 52, 31, 15, 73, 93, 29, 55, 79, 95, 37, 62, 35, 20]
        val_subjects = [42, 84, 85, 44, 81, 56, 75, 78, 1, 61, 74, 68, 22, 64, 91, 71]
    
    elif setting == '64vs4_TestBucket8':
        test_subjects = [45, 24, 36, 71]
        train_subjects = [55, 21, 74, 28, 76, 91, 20, 93, 95, 29, 35, 34, 32, 68, 27, 40, 84, 5, 82, 38, 47, 78, 97, 75, 56, 7, 85, 62, 37, 79, 58, 72, 13, 15, 25, 42, 64, 43, 48, 44, 80, 31, 69, 70, 92, 60, 86, 46]
        val_subjects = [54, 22, 23, 65, 49, 94, 83, 81, 14, 61, 57, 73, 52, 1, 51, 63]
    
    elif setting == '64vs4_TestBucket9':
        test_subjects = [91, 85, 61, 83]
        train_subjects = [31, 60, 81, 80, 82, 1, 54, 97, 62, 45, 24, 92, 48, 74, 93, 20, 63, 84, 37, 55, 49, 7, 76, 23, 25, 40, 69, 27, 64, 43, 47, 79, 32, 14, 38, 72, 68, 65, 35, 70, 28, 13, 75, 15, 73, 5, 71, 36]
        val_subjects = [46, 34, 21, 29, 95, 51, 57, 56, 78, 94, 52, 22, 86, 58, 44, 42]
    
    elif setting == '64vs4_TestBucket10':
        test_subjects = [94, 31, 43, 54]
        train_subjects = [52, 74, 37, 25, 70, 95, 23, 47, 85, 63, 76, 7, 32, 58, 1, 14, 91, 55, 71, 83, 79, 34, 62, 40, 69, 64, 73, 27, 24, 22, 84, 13, 78, 48, 80, 92, 61, 72, 21, 29, 82, 86, 36, 51, 42, 75, 35, 56]
        val_subjects = [97, 93, 68, 45, 49, 15, 81, 65, 20, 57, 44, 60, 38, 5, 46, 28]
        
    elif setting == '64vs4_TestBucket11':
        test_subjects = [51, 64, 68, 44]
        train_subjects = [38, 61, 15, 79, 97, 36, 78, 57, 35, 28, 73, 75, 29, 43, 63, 85, 95, 86, 76, 5, 70, 58, 60, 46, 80, 65, 42, 34, 24, 21, 37, 47, 20, 32, 81, 45, 62, 56, 93, 14, 13, 49, 52, 94, 48, 55, 23, 54]
        val_subjects = [84, 91, 1, 74, 40, 7, 83, 82, 31, 27, 25, 71, 92, 72, 22, 69]
    
    elif setting == '64vs4_TestBucket12':
        test_subjects = [20, 32, 5, 49]
        train_subjects = [61, 46, 63, 79, 24, 29, 60, 57, 69, 71, 72, 43, 62, 82, 54, 45, 84, 85, 15, 65, 1, 25, 23, 73, 42, 48, 81, 7, 13, 37, 55, 22, 91, 51, 95, 38, 76, 58, 14, 75, 83, 28, 44, 64, 68, 31, 97, 52]
        val_subjects = [92, 36, 93, 47, 80, 74, 34, 86, 27, 56, 70, 78, 21, 40, 94, 35]
    
    elif setting == '64vs4_TestBucket13':
        test_subjects = [65, 28, 78, 37]
        train_subjects = [62, 84, 47, 31, 15, 42, 43, 40, 79, 48, 38, 80, 97, 45, 14, 85, 55, 75, 76, 44, 69, 29, 51, 95, 5, 25, 71, 46, 92, 56, 52, 83, 72, 61, 70, 22, 81, 74, 68, 27, 34, 94, 58, 20, 91, 63, 93, 49]
        val_subjects = [21, 86, 82, 35, 24, 13, 32, 60, 54, 64, 57, 36, 23, 7, 1, 73]
    
    elif setting == '64vs4_TestBucket14':
        test_subjects = [97, 40, 74, 46]
        train_subjects = [68, 95, 58, 1, 63, 51, 43, 85, 48, 81, 21, 86, 55, 54, 28, 7, 52, 91, 84, 22, 60, 73, 25, 20, 62, 45, 36, 70, 24, 38, 5, 29, 42, 27, 34, 44, 37, 93, 64, 72, 76, 69, 80, 47, 71, 65, 78, 13]
        val_subjects = [92, 75, 83, 35, 15, 56, 57, 61, 79, 23, 31, 82, 14, 32, 49, 94]
    
    elif setting == '64vs4_TestBucket15':
        test_subjects = [22, 7, 23, 95]
        train_subjects = [85, 70, 27, 78, 49, 24, 57, 81, 94, 65, 51, 34, 1, 54, 61, 92, 63, 68, 47, 73, 91, 80, 58, 97, 15, 45, 28, 79, 74, 83, 64, 32, 36, 14, 71, 13, 46, 84, 60, 43, 52, 48, 44, 5, 93, 69, 76, 75]
        val_subjects = [21, 56, 29, 55, 38, 82, 62, 40, 20, 37, 35, 42, 72, 31, 86, 25]
    
    elif setting == '64vs4_TestBucket16':
        test_subjects = [13, 35, 1, 34]
        train_subjects = [47, 83, 60, 81, 29, 76, 61, 94, 5, 51, 42, 48, 7, 27, 58, 21, 45, 32, 36, 92, 46, 31, 62, 57, 86, 73, 71, 84, 80, 65, 79, 64, 69, 56, 44, 68, 20, 23, 43, 54, 72, 74, 37, 93, 38, 97, 75, 78]
        val_subjects = [40, 95, 85, 52, 63, 28, 25, 15, 49, 70, 91, 24, 55, 14, 22, 82]
    
    elif setting == '64vs4_TestBucket17':
        test_subjects = [21, 25, 29, 60]
        train_subjects = [28, 36, 52, 43, 22, 44, 72, 15, 79, 75, 85, 37, 32, 38, 45, 63, 14, 97, 83, 31, 80, 73, 70, 24, 13, 20, 1, 94, 68, 93, 61, 86, 46, 64, 65, 91, 84, 54, 69, 81, 78, 55, 74, 35, 40, 49, 27, 42]
        val_subjects = [48, 56, 34, 57, 95, 62, 76, 23, 51, 47, 58, 5, 7, 92, 82, 71]
    
    elif setting == '16vs4_TestBucket1':
        test_subjects = [86, 56, 72, 79]
        train_subjects = [85, 29, 44, 25, 74, 22, 40, 75, 64, 65, 7, 76]
        val_subjects = [21, 78, 55, 62]
    
    elif setting == '16vs4_TestBucket2':
        test_subjects = [93, 82, 55, 48]
        train_subjects = [23, 28, 91, 65, 63, 56, 35, 73, 15, 20, 7, 49]
        val_subjects = [21, 86, 25, 70]
    
    elif setting == '16vs4_TestBucket3':
        test_subjects = [80, 14, 58, 75]
        train_subjects = [82, 56, 36, 72, 44, 84, 49, 63, 43, 27, 24, 69]
        val_subjects = [74, 15, 94, 22]
    
    elif setting == '16vs4_TestBucket4':
        test_subjects = [62, 47, 52, 84]
        train_subjects = [71, 34, 82, 92, 49, 60, 81, 75, 57, 46, 86, 70]
        val_subjects = [74, 51, 38, 37]
    
    elif setting == '16vs4_TestBucket5':
        test_subjects = [73, 69, 42, 63]
        train_subjects = [40, 62, 81, 27, 93, 47, 48, 97, 57, 85, 64, 60]
        val_subjects = [20, 1, 68, 24]
    
    elif setting == '16vs4_TestBucket6':
        test_subjects = [81, 15, 57, 70]
        train_subjects = [38, 13, 1, 95, 32, 68, 71, 84, 22, 43, 58, 62]
        val_subjects = [92, 47, 74, 82]
    
    elif setting == '16vs4_TestBucket7':
        test_subjects = [27, 92, 38, 76]
        train_subjects = [85, 32, 80, 91, 71, 14, 1, 49, 24, 78, 35, 34]
        val_subjects = [28, 47, 64, 94]
    
    elif setting == '16vs4_TestBucket8':
        test_subjects = [45, 24, 36, 71]
        train_subjects = [13, 76, 44, 72, 69, 70, 68, 34, 21, 84, 15, 82]
        val_subjects = [55, 97, 60, 83]
    
    elif setting == '16vs4_TestBucket9':
        test_subjects = [91, 85, 61, 83]
        train_subjects = [69, 36, 29, 31, 75, 84, 32, 28, 37, 54, 43, 49]
        val_subjects = [38, 79, 52, 14]
    
    elif setting == '16vs4_TestBucket10':
        test_subjects = [94, 31, 43, 54]
        train_subjects = [52, 58, 42, 80, 72, 68, 93, 56, 95, 44, 63, 64]
        val_subjects = [5, 14, 79, 81]
    
    elif setting == '16vs4_TestBucket11':
        test_subjects = [51, 64, 68, 44]
        train_subjects = [58, 49, 21, 93, 62, 37, 32, 71, 56, 73, 82, 97]
        val_subjects = [36, 61, 13, 45]
    
    elif setting == '16vs4_TestBucket12':
        test_subjects = [20, 32, 5, 49]
        train_subjects = [69, 40, 14, 44, 58, 37, 60, 85, 64, 68, 65, 61]
        val_subjects = [47, 76, 28, 55]
    
    elif setting == '16vs4_TestBucket13':
        test_subjects = [65, 28, 78, 37]
        train_subjects = [56, 62, 20, 72, 80, 54, 64, 70, 22, 35, 58, 74]
        val_subjects = [57, 5, 34, 43]
    
    elif setting == '16vs4_TestBucket14':
        test_subjects = [97, 40, 74, 46]
        train_subjects = [15, 34, 36, 32, 52, 42, 91, 21, 37, 20, 48, 81]
        val_subjects = [83, 69, 84, 64]
    
    elif setting == '16vs4_TestBucket15':
        test_subjects = [22, 7, 23, 95]
        train_subjects = [84, 40, 60, 48, 57, 13, 69, 37, 32, 55, 56, 21]
        val_subjects = [76, 80, 85, 75]
    
    elif setting == '16vs4_TestBucket16':
        test_subjects = [13, 35, 1, 34]
        train_subjects = [92, 29, 78, 72, 85, 80, 70, 95, 15, 24, 7, 97]
        val_subjects = [81, 52, 69, 63]
    
    elif setting == '16vs4_TestBucket17':
        test_subjects = [21, 25, 29, 60]
        train_subjects = [38, 1, 27, 62, 83, 84, 70, 24, 20, 5, 85, 52]
        val_subjects = [76, 34, 63, 86]
    
    elif setting == '4vs4_TestBucket1':
        test_subjects = [86, 56, 72, 79]
        train_subjects = [37, 76, 83]
        val_subjects = [31]
    
    elif setting == '4vs4_TestBucket2':
        test_subjects = [93, 82, 55, 48]
        train_subjects = [38, 78, 54]
        val_subjects = [47]
    
    elif setting == '4vs4_TestBucket3':
        test_subjects = [80, 14, 58, 75]
        train_subjects = [44, 82, 97]
        val_subjects = [29]
    
    elif setting == '4vs4_TestBucket4':
        test_subjects = [62, 47, 52, 84]
        train_subjects = [42, 55, 34]
        val_subjects = [46]
    
    elif setting == '4vs4_TestBucket5':
        test_subjects = [73, 69, 42, 63]
        train_subjects = [92, 60, 21]
        val_subjects = [38]
    
    elif setting == '4vs4_TestBucket6':
        test_subjects = [81, 15, 57, 70]
        train_subjects = [44, 97, 37]
        val_subjects = [72]
    
    elif setting == '4vs4_TestBucket7':
        test_subjects = [27, 92, 38, 76]
        train_subjects = [7, 14, 63]
        val_subjects = [64]
    
    elif setting == '4vs4_TestBucket8':
        test_subjects = [45, 24, 36, 71]
        train_subjects = [20, 72, 65]
        val_subjects = [58]
    
    elif setting == '4vs4_TestBucket9':
        test_subjects = [91, 85, 61, 83]
        train_subjects = [13, 47, 54]
        val_subjects = [95]
    
    elif setting == '4vs4_TestBucket10':
        test_subjects = [94, 31, 43, 54]
        train_subjects = [79, 97, 40]
        val_subjects = [82]
    
    elif setting == '4vs4_TestBucket11':
        test_subjects = [51, 64, 68, 44]
        train_subjects = [81, 80, 5]
        val_subjects = [71]
    
    elif setting == '4vs4_TestBucket12':
        test_subjects = [20, 32, 5, 49]
        train_subjects = [37, 14, 71]
        val_subjects = [52]
    
    elif setting == '4vs4_TestBucket13':
        test_subjects = [65, 28, 78, 37]
        train_subjects = [63, 32, 84]
        val_subjects = [31]
    
    elif setting == '4vs4_TestBucket14':
        test_subjects = [97, 40, 74, 46]
        train_subjects = [71, 63, 45]
        val_subjects = [81]
    
    elif setting == '4vs4_TestBucket15':
        test_subjects = [22, 7, 23, 95]
        train_subjects = [54, 80, 57]
        val_subjects = [55]
    
    elif setting == '4vs4_TestBucket16':
        test_subjects = [13, 35, 1, 34]
        train_subjects = [82, 86, 15]
        val_subjects = [61]
    
    elif setting == '4vs4_TestBucket17':
        test_subjects = [21, 25, 29, 60]
        train_subjects = [80, 85, 69]
        val_subjects = [23]
    else:
        raise NameError('not supported setting')
    
    #sanity check:
    print('data_dir: {}, type: {}'.format(data_dir, type(data_dir)))
    print('window_size: {}, type: {}'.format(window_size, type(window_size)))
    print('result_save_rootdir: {}, type: {}'.format(result_save_rootdir, type(result_save_rootdir)))
    print('classification_task: {}, type: {}'.format(classification_task, type(classification_task)))
    print('restore_file: {} type: {}'.format(restore_file, type(restore_file)))
    print('n_epoch: {} type: {}'.format(n_epoch, type(n_epoch)))
    print('setting: {} type: {}'.format(setting, type(setting)))
    
    args_dict = edict() 
    
    args_dict.gpu_idx = gpu_idx
    args_dict.data_dir = data_dir
    args_dict.window_size = window_size
    args_dict.result_save_rootdir = result_save_rootdir
    args_dict.classification_task = classification_task
    args_dict.restore_file = restore_file
    args_dict.n_epoch = n_epoch

    
    
    seed_everything(seed)
    train_classifier(args_dict, train_subjects, val_subjects, test_subjects)
    
