import os
import sys
import numpy as np
import torch
import torch.nn as nn

import argparse

from easydict import EasyDict as edict
from tqdm import trange

sys.path.insert(0, '/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/helpers/')
import models
import brain_data
from utils import seed_everything, makedir_if_not_exist, plot_confusion_matrix, save_pickle, train_one_epoch, eval_model, save_training_curves_FixedTrainValSplit, write_performance_info_FixedTrainValSplit

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
parser.add_argument('--setting', default='seed1', help='which predefined train val test split scenario')


#for personal model, save the test prediction of each cv fold
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
    gpu_idx = args_dict.gpu_idx
    data_dir = args_dict.data_dir
    window_size = args_dict.window_size
    result_save_rootdir = args_dict.result_save_rootdir
    classification_task = args_dict.classification_task
    restore_file = args_dict.restore_file
    n_epoch = args_dict.n_epoch
    
    model_to_use = models.EEGNet150
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
                        
                        test_accuracy, test_class_predictions, test_class_labels, test_logits = eval_model(model, test_subjects_dict[test_subject]['sub_test_loader'], device)
                        
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

    if setting == 'seed1':
        train_subjects = [40, 95, 31, 80, 42, 85, 15, 32, 45, 29, 97, 86, 82, 23, 92, 20]
        val_subjects = [48, 36, 14, 64, 34]
        test_subjects_URG = [22, 70, 78, 28, 60, 58]
        test_subjects_WHITE = [69, 47, 79, 21, 91, 38]
        test_subjects_ASIAN = [94, 13, 56, 76, 93, 27]
        
    elif setting == 'seed2':
        train_subjects = [38, 45, 21, 31, 48, 14, 34, 91, 42, 29, 20, 85, 36, 23, 86, 79]
        val_subjects = [32, 95, 40, 82, 47]
        test_subjects_URG = [22, 70, 78, 28, 60, 58]
        test_subjects_WHITE = [64, 69, 80, 92, 97, 15]
        test_subjects_ASIAN = [25, 7, 54, 24, 37, 94]
    
    elif setting == 'seed3':
        train_subjects = [97, 92, 38, 47, 48, 32, 69, 45, 15, 64, 91, 79, 95, 42, 14, 31]
        val_subjects = [86, 40, 21, 80, 23]
        test_subjects_URG = [22, 70, 78, 28, 60, 58]
        test_subjects_WHITE = [36, 85, 34, 82, 29, 20]
        test_subjects_ASIAN = [55, 76, 56, 24, 13, 93]

    elif setting == 'seed4':
        train_subjects = [29, 40, 31, 32, 69, 20, 14, 23, 95, 15, 45, 38, 42, 97, 79, 64]
        val_subjects = [92, 34, 47, 85, 80]
        test_subjects_URG = [22, 70, 78, 28, 60, 58]
        test_subjects_WHITE = [21, 91, 86, 36, 82, 48]
        test_subjects_ASIAN = [84, 93, 13, 37, 5, 51]
    
    elif setting == 'seed5':
        train_subjects = [92, 36, 14, 21, 64, 47, 42, 32, 91, 85, 15, 45, 38, 80, 95, 23]
        val_subjects = [29, 40, 31, 82, 48]
        test_subjects_URG = [22, 70, 78, 28, 60, 58]
        test_subjects_WHITE = [34, 97, 86, 20, 79, 69]
        test_subjects_ASIAN = [49, 57, 43, 7, 56, 61]

    elif setting == 'seed6':
        train_subjects = [36, 20, 64, 85, 95, 45, 29, 21, 79, 15, 32, 86, 38, 82, 97, 42]
        val_subjects = [80, 14, 47, 48, 23]
        test_subjects_URG = [22, 70, 78, 28, 60, 58]
        test_subjects_WHITE = [92, 34, 69, 40, 91, 31]
        test_subjects_ASIAN = [49, 93, 72, 71, 75, 63]
    
    elif setting == 'seed7':
        train_subjects = [21, 47, 92, 40, 36, 97, 48, 20, 91, 38, 82, 64, 23, 42, 79, 95]
        val_subjects = [80, 29, 15, 45, 14]
        test_subjects_URG = [22, 70, 78, 28, 60, 58]
        test_subjects_WHITE = [69, 85, 34, 31, 86, 32]
        test_subjects_ASIAN = [51, 25, 44, 65, 52, 56]
    
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
    train_classifier(args_dict, train_subjects, val_subjects, test_subjects_URG, test_subjects_WHITE, test_subjects_ASIAN)
    
