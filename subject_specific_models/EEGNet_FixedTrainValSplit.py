import os
import sys
import numpy as np
import torch
import torch.nn as nn

import time
import argparse

from easydict import EasyDict as edict
from tqdm import trange

sys.path.insert(0, '/cluster/tufts/hugheslab/zhuang12/HCI/NuripsDataSet2021/helpers/')
import models
import brain_data
from utils import seed_everything, makedir_if_not_exist, plot_confusion_matrix, save_pickle, train_one_epoch, eval_model, save_training_curves_FixedTrainValSplit, write_performance_info_FixedTrainValSplit, write_program_time

from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int, help="random seed")
parser.add_argument('--gpu_idx', default=0, type=int, help="gpu idx")
parser.add_argument('--data_dir', default='../data/Leon/Visual/size_40sec_200ts_stride_3ts/', help="folder to the dataset")
parser.add_argument('--window_size', default=200, type=int, help='window size')
parser.add_argument('--result_save_rootdir', default='./experiments', help="Directory containing the dataset")
parser.add_argument('--SubjectId_of_interest', default='1', help="training personal model for which subject")
parser.add_argument('--classification_task', default='four_class', help='binary or four-class classification')
parser.add_argument('--restore_file', default='None', help="xxx.statedict")

# #fixed hyper for 40sec binary classification tasl
# parser.add_argument('--cv_train_batch_size', default=243, type=int, help="cross validation train batch size")
# parser.add_argument('--cv_val_batch_size', default=61, type=int, help="cross validation val batch size")
# parser.add_argument('--test_batch_size', default=304, type=int, help="test batch size")
parser.add_argument('--n_epoch', default=100, type=int, help="number of epoch")



#for personal model, save the test prediction of each cv fold
def train_classifier(args_dict):
    
    #parse args:
    gpu_idx = args_dict.gpu_idx
    data_dir = args_dict.data_dir
    window_size = args_dict.window_size
    result_save_rootdir = args_dict.result_save_rootdir
    SubjectId_of_interest = args_dict.SubjectId_of_interest
    classification_task = args_dict.classification_task
    restore_file = args_dict.restore_file
    
#     cv_train_batch_size = args_dict.cv_train_batch_size 
#     cv_val_batch_size = args_dict.cv_val_batch_size
#     test_batch_size = args_dict.test_batch_size 
    n_epoch = args_dict.n_epoch
    

    
    #load this subject's data
    sub_file = 'sub_{}.csv'.format(SubjectId_of_interest)
    
    if window_size == 10:
        model_to_use = models.EEGNet10
        num_chunk_this_window_size = 2224
    elif window_size == 25:
        model_to_use = models.EEGNet25
        num_chunk_this_window_size = 2144
    elif window_size == 50:
        model_to_use = models.EEGNet50
        num_chunk_this_window_size = 2016
    elif window_size == 100:
        model_to_use = models.EEGNet100
        num_chunk_this_window_size = 1744
    elif window_size == 150:
        model_to_use = models.EEGNet150
        num_chunk_this_window_size = 1488
    elif window_size == 200:
        model_to_use = models.EEGNet
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
        
   
    #load the subject's data
    sub_feature_array, sub_label_array = data_loading_function(os.path.join(data_dir, sub_file),  num_chunk_this_window_size=num_chunk_this_window_size)
    
    sub_data_len = len(sub_label_array)
    #use 1st half as train, 2nd half as test
    half_sub_data_len = int(sub_data_len/2)
    print('half_sub_data_len: {}'.format(half_sub_data_len), flush=True)
    
    sub_train_feature_array = sub_feature_array[:half_sub_data_len]
    sub_train_label_array = sub_label_array[:half_sub_data_len]

    sub_test_feature_array = sub_feature_array[half_sub_data_len:]
    sub_test_label_array = sub_label_array[half_sub_data_len:]

    #convert subject's test data into dataset object
    sub_test_set = brain_data.brain_dataset(sub_test_feature_array, sub_test_label_array)
    test_batch_size = len(sub_test_set)
    
    #convert subject's test dataset object into dataloader object
    sub_test_loader = torch.utils.data.DataLoader(sub_test_set, batch_size=test_batch_size, shuffle=False)
  
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

            #Mar21: Control: Do not rerun already finished experiment:
            #(if the result_analysis/performance.txt already exist, meaning this experiment has already finished previously)
            AlreadyFinished = os.path.exists(os.path.join(result_save_rootdir, SubjectId_of_interest, experiment_name, 'result_analysis', 'performance.txt'))
            if AlreadyFinished:
                print('{}, lr:{} dropout:{} already finished, Do Not Rerun, continue to next setting'.format(SubjectId_of_interest, lr, dropout), flush = True)
                continue

            #derived arg
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
#                     train_index = total_index[:228]
#                     val_index = total_index[244:]
                    train_index = total_index[:152]
                    val_index = total_index[152:]
        
                elif window_size == 150:
                    total_number_train_chunks = 368
                    total_index = np.arange(total_number_train_chunks)
#                     train_index = total_index[:276]
#                     val_index = total_index[295:]
                    train_index = total_index[:184]
                    val_index = total_index[184:]

                elif window_size == 100:
                    total_number_train_chunks = 436
                    total_index = np.arange(total_number_train_chunks)
#                     train_index = total_index[:327]
#                     val_index = total_index[349:]
                    train_index = total_index[:218]
                    val_index = total_index[218:]
                    
                elif window_size == 50:
                    total_number_train_chunks = 504
                    total_index = np.arange(total_number_train_chunks)
#                     train_index = total_index[:388]
#                     val_index = total_index[404:]
                    train_index = total_index[:252]
                    val_index = total_index[252:]
    
                elif window_size == 25:
                    total_number_train_chunks = 536
                    total_index = np.arange(total_number_train_chunks)
#                     train_index = total_index[:422]
#                     val_index = total_index[429:]
                    train_index = total_index[:268]
                    val_index = total_index[268:]
        
                elif window_size == 10:
                    total_number_train_chunks = 556
                    total_index = np.arange(total_number_train_chunks)
#                     train_index = total_index[:443]
#                     val_index = total_index[445:]
                    train_index = total_index[:278]
                    val_index = total_index[278:]
        
                else:
                    raise NameError('not supported window size') 
            else:
                raise NameError('not implemented classification task')
            
            
            #dataset object
            sub_cv_train_set = brain_data.brain_dataset(sub_train_feature_array[train_index], sub_train_label_array[train_index])
            sub_cv_val_set = brain_data.brain_dataset(sub_train_feature_array[val_index], sub_train_label_array[val_index])

            #dataloader object
            cv_train_batch_size = len(sub_cv_train_set)
            cv_val_batch_size = len(sub_cv_val_set)
            sub_cv_train_loader = torch.utils.data.DataLoader(sub_cv_train_set, batch_size=cv_train_batch_size, shuffle=True) 
            sub_cv_val_loader = torch.utils.data.DataLoader(sub_cv_val_set, batch_size=cv_val_batch_size, shuffle=False)

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
                average_loss_this_epoch = train_one_epoch(model, optimizer, criterion, sub_cv_train_loader, device)
                val_accuracy, _, _, _ = eval_model(model, sub_cv_val_loader, device)
                train_accuracy, _, _ , _ = eval_model(model, sub_cv_train_loader, device)

                epoch_train_loss.append(average_loss_this_epoch)
                epoch_train_accuracy.append(train_accuracy)
                epoch_validation_accuracy.append(val_accuracy)

                #update is_best flag
                is_best = val_accuracy >= best_val_accuracy

                if is_best:
                    best_val_accuracy = val_accuracy

                    torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'best_model.statedict'))

                    #in the script use the name "logits" (what we mean in the code is score after log-softmax normalization) and "probabilities" interchangibly 
                    test_accuracy, test_class_predictions, test_class_labels, test_logits = eval_model(model, sub_test_loader, device)
                    print('test accuracy at this epoch is {}'.format(test_accuracy))

                    result_save_dict['bestepoch_test_accuracy'] = test_accuracy
                    result_save_dict['bestepoch_val_accuracy'] = val_accuracy

                    result_save_dict['bestepoch_test_logits'] = test_logits.copy()
                    result_save_dict['bestepoch_test_class_labels'] = test_class_labels.copy()


            #save training curve 
            save_training_curves_FixedTrainValSplit('training_curve.png', result_save_subject_trainingcurvedir, epoch_train_loss, epoch_train_accuracy, epoch_validation_accuracy)

            #confusion matrix 
            plot_confusion_matrix(test_class_predictions, test_class_labels, confusion_matrix_figure_labels, result_save_subject_resultanalysisdir, 'test_confusion_matrix.png')

            #save the model at last epoch
            torch.save(model.state_dict(), os.path.join(result_save_subject_checkpointdir, 'last_model.statedict'))


            #save result_save_dict
            save_pickle(result_save_subject_predictionsdir, 'result_save_dict.pkl', result_save_dict)
            
            #write performance to txt file
            write_performance_info_FixedTrainValSplit(model.state_dict(), result_save_subject_resultanalysisdir, result_save_dict['bestepoch_val_accuracy'], result_save_dict['bestepoch_test_accuracy'])
    
    end_time = time.time()
    total_time = end_time - start_time
    write_program_time(os.path.join(result_save_rootdir, SubjectId_of_interest), total_time)
    



if __name__=='__main__':
    
    #parse args
    args = parser.parse_args()
    
    seed = args.seed
    gpu_idx = args.gpu_idx
    data_dir = args.data_dir
    window_size = args.window_size
    result_save_rootdir = args.result_save_rootdir
    SubjectId_of_interest = args.SubjectId_of_interest
    classification_task = args.classification_task
    restore_file = args.restore_file
    
#     cv_train_batch_size = args.cv_train_batch_size
#     cv_val_batch_size = args.cv_val_batch_size
#     test_batch_size = args.test_batch_size
    n_epoch = args.n_epoch

    
    #sanity check:
    print('type(data_dir): {}'.format(type(data_dir)))
    print('type(window_size): {}'.format(type(window_size)))
    print('type(result_save_rootdir): {}'.format(type(result_save_rootdir)))
    print('type(SubjectId_of_interest): {}'.format(type(SubjectId_of_interest)))
    print('type(classification_task): {}'.format(type(classification_task)))
    print('type(restore_file): {}'.format(type(restore_file)))
    print('type(n_epoch): {}'.format(type(n_epoch)))
       
    
    args_dict = edict() 
    
    args_dict.gpu_idx = gpu_idx
    args_dict.data_dir = data_dir
    args_dict.window_size = window_size
    args_dict.result_save_rootdir = result_save_rootdir
    args_dict.SubjectId_of_interest = SubjectId_of_interest
    args_dict.classification_task = classification_task
    args_dict.restore_file = restore_file
#     args_dict.cv_train_batch_size = cv_train_batch_size
#     args_dict.cv_val_batch_size = cv_val_batch_size
#     args_dict.test_batch_size = test_batch_size
    args_dict.n_epoch = n_epoch

    
    
    seed_everything(seed)
    train_classifier(args_dict)
    
