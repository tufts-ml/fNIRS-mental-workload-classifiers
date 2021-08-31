import pickle
import time
import numpy as np
import torch
import csv 
import os
import random
import logging
import shutil
import torch.nn.functional as F

from matplotlib import gridspec
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix as sklearn_cm
import seaborn as sns

def load_pickle(result_dir, filename):
    with open(os.path.join(result_dir, filename), 'rb') as f:
        data = pickle.load(f)
    
    return data


def save_pickle(save_dir, save_file_name, data):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data_save_fullpath = os.path.join(save_dir, save_file_name)
    with open(data_save_fullpath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def makedir_if_not_exist(specified_dir):
    if not os.path.exists(specified_dir):
        os.makedirs(specified_dir)

        
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

#Mar23
def get_slope_and_intercept(column_values, return_value = 'w'):
    
    num_timesteps = len(column_values)
    print('num_timesteps: {}'.format(num_timesteps))
    tvec_T = np.linspace(0, 1, num_timesteps) #already asserted len(column_values) = 10
    tdiff_T = tvec_T - np.mean(tvec_T)
    
    w = np.inner(column_values - np.mean(column_values), tdiff_T) / np.sum(np.square(tdiff_T))
    b = np.mean(column_values) - w * np.mean(tvec_T)
    
    if return_value == 'w':
        return w
    
    elif return_value == 'b':
        return b
    
    else:
        raise Exception("invalid return_value")
        
        
def featurize(sub_feature_array, classification_task='four_class'):
    
    num_data = sub_feature_array.shape[0]
    num_features = sub_feature_array.shape[2]
    
    assert num_features == 8 #8 features
    
    transformed_sub_feature_array = []
    for i in range(num_data):
        this_chunk_data = sub_feature_array[i]
        this_chunk_column_means = np.mean(this_chunk_data, axis=0)
        this_chunk_column_stds = np.std(this_chunk_data, axis=0)
        this_chunk_column_slopes = np.array([get_slope_and_intercept(this_chunk_data[:,i], 'w') for i in range(num_features)])
        this_chunk_column_intercepts = np.array([get_slope_and_intercept(this_chunk_data[:,i], 'b') for i in range(num_features)])
        
        this_chunk_transformed_features = np.concatenate([this_chunk_column_means, this_chunk_column_stds, this_chunk_column_slopes, this_chunk_column_intercepts])
        
        transformed_sub_feature_array.append(this_chunk_transformed_features)
    
    return np.array(transformed_sub_feature_array)


def plot_confusion_matrix(predictions, true_labels, figure_labels, save_dir, filename):
    
    sns.set(color_codes=True)
    sns.set(font_scale=1.4)
    
    plt.figure(1, figsize=(8,5))
    plt.title('Confusion Matrix')
    
    data = sklearn_cm(true_labels, predictions)
    ax = sns.heatmap(data, annot=True, fmt='d', cmap='Blues')
    
    ax.set_xticklabels(figure_labels)
    ax.set_yticklabels(figure_labels)
    ax.set(ylabel='True Label', xlabel='Predicted Label')
    ax.set_ylim([4, 0])
    
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
    

def save_training_curves_FixedTrainValSplit(figure_name, result_save_subject_trainingcurvedir, epoch_train_loss, epoch_train_accuracy=None, epoch_validation_accuracy = None, epoch_test_accuracy = None):
    
    fig = plt.figure(figsize=(15, 8))
    
    ax_1 = fig.add_subplot(1,4,1)
    ax_1.plot(range(len(epoch_train_loss)), epoch_train_loss, label='epoch_train_loss')
    
    if epoch_train_accuracy is not None:
        ax_2 = fig.add_subplot(1,4,2, sharex = ax_1)
        ax_2.plot(range(len(epoch_train_accuracy)), epoch_train_accuracy, label='epoch_train_accuracy')
        ax_2.legend()
        
    if epoch_validation_accuracy is not None:
        ax_3 = fig.add_subplot(1,4,3, sharex = ax_1)
        ax_3.plot(range(len(epoch_validation_accuracy)), epoch_validation_accuracy, label='epoch_validation_accuracy')
        ax_3.legend()
    
    if epoch_test_accuracy is not None:
        ax_4 = fig.add_subplot(1,4,4)
        ax_4.plot(range(len(epoch_test_accuracy)), epoch_test_accuracy, label='epoch_test_accuracy')
        ax_4.legend()
    
    ax_1.legend()
        
    figure_save_path = os.path.join(result_save_subject_trainingcurvedir, figure_name)
    plt.savefig(figure_save_path)
    plt.close()
    

def save_training_curves_FixedTrainValSplit_overlaid(figure_name, result_save_subject_trainingcurvedir, epoch_train_loss, epoch_train_accuracy=None, epoch_validation_accuracy = None, epoch_test_accuracy = None):
    
    fig = plt.figure(figsize=(15, 8))
    
    ax_1 = fig.add_subplot(1,2,1)
    ax_1.plot(range(len(epoch_train_loss)), epoch_train_loss, label='epoch_train_loss')
    
    ax_2 = fig.add_subplot(1,2,2)
    ax_2.plot(range(len(epoch_train_accuracy)), epoch_train_accuracy, label='epoch_train_accuracy')
    ax_2.plot(range(len(epoch_validation_accuracy)), epoch_validation_accuracy, label='epoch_validation_accuracy')
    ax_2.plot(range(len(epoch_test_accuracy)), epoch_test_accuracy, label='epoch_test_accuracy')

    ax_2.legend()
    
    ax_1.legend()
        
    figure_save_path = os.path.join(result_save_subject_trainingcurvedir, figure_name)
    plt.savefig(figure_save_path)
    plt.close()
    
    

#Aug19
def write_performance_info_FixedTrainValSplit(model_state_dict, result_save_subject_resultanalysisdir, highest_validation_accuracy, corresponding_test_accuracy):
    #create file writer
    file_writer = open(os.path.join(result_save_subject_resultanalysisdir, 'performance.txt'), 'w')
    
    #write performance to file
    file_writer.write('highest validation accuracy: {}\n'.format(highest_validation_accuracy))
    file_writer.write('corresponding test accuracy: {}\n'.format(corresponding_test_accuracy))
    #write model parameters to file
    file_writer.write('Model parameters:\n')
    
    if model_state_dict != 'NA':
        total_elements = 0
        for name, tensor in model_state_dict.items():
            file_writer.write('layer {}: {} parameters\n'.format(name, torch.numel(tensor)))
            total_elements += torch.numel(tensor)
        file_writer.write('total elemets in this model: {}'.format(total_elements))
    else:
        file_writer.write('total elemets in this model NA, sklearn model')
    
    file_writer.close()
    
def write_initial_test_accuracy(result_save_subject_resultanalysisdir, initial_test_accuracy):
    #create file writer
    file_writer = open(os.path.join(result_save_subject_resultanalysisdir, 'initial_test_accuracy.txt'), 'w')
    
    #write performance to file
    file_writer.write('initial test accuracy: {}\n'.format(initial_test_accuracy))
    
    file_writer.close()

def write_program_time(result_save_subject_resultanalysisdir, time_in_seconds):
    #create file writer
    file_writer = open(os.path.join(result_save_subject_resultanalysisdir, 'program_time.txt'), 'w')
    
    #write performance to file
    file_writer.write('program_time: {} seconds \n'.format(round(time_in_seconds,2)))
    
    file_writer.close()

def write_inference_time(result_save_subject_resultanalysisdir, time_in_seconds):
    #create file writer
    file_writer = open(os.path.join(result_save_subject_resultanalysisdir, 'inference_time.txt'), 'w')
    
    #write performance to file
    file_writer.write('program_time: {} seconds \n'.format(round(time_in_seconds,2)))
    
    file_writer.close()

    
#Aug13
def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    
    loss_avg = RunningAverage()
    for i, (data_batch, labels_batch) in enumerate(train_loader):
#         print('Inside train_one_epoch, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features])
        #labels: tensor on cpu, torch.Size([batch_size])
        
        data_batch = data_batch.to(device) #put inputs to device
        labels_batch = labels_batch.to(device) #when performing training, need to also put labels to device to do loss calculation and backpropagation

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch = model(data_batch)
        
        #calculate loss
        #loss: tensor (scalar) on gpu, torch.Size([])
        loss = criterion(output_batch, labels_batch)
        
        #update running average of the loss
        loss_avg.update(loss.item())
        
        #clear previous gradients
        optimizer.zero_grad()

        #calculate gradient
        loss.backward()
        #perform parameters update
        optimizer.step()
    
    average_loss_this_epoch = loss_avg()
    return average_loss_this_epoch


def eval_model(model, eval_loader, device):
    
    #reference: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/evaluate.py
    #set the model to evaluation mode
    model.eval()
    
#     predicted_array = None # 1d numpy array, [batch_size * num_batches]
    labels_array = None # 1d numpy array, [batch_size * num_batches]
    probabilities_array = None # 2d numpy array, [batch_size * num_batches, num_classes] 
    
    for data_batch, labels_batch in eval_loader:#test_loader
        print('Inside eval_model, size of data_batch is {}'.format(data_batch.shape))
        #inputs: tensor on cpu, torch.Size([batch_size, sequence_length, num_features])
        #labels: tensor on cpu, torch.Size([batch_size])
       
        data_batch = data_batch.to(device) #put inputs to device

        #forward pass
        #outputs: tensor on gpu, requires grad, torch.Size([batch_size, num_classes])
        output_batch = model(data_batch)
        
        #extract data from torch variable, move to cpu, convert to numpy arrays    
        if labels_array is None:
#             label_array = labels.numpy()
            labels_array = labels_batch.data.cpu().numpy()
            
        else:
            labels_array = np.concatenate((labels_array, labels_batch.data.cpu().numpy()), axis=0)#np.concatenate without axis will flattened to 1d array
        
        
        if probabilities_array is None:
            probabilities_array = output_batch.data.cpu().numpy()
        else:
            probabilities_array = np.concatenate((probabilities_array, output_batch.data.cpu().numpy()), axis = 0) #concatenate on batch dimension: torch.Size([batch_size * num_batches, num_classes])
            
    class_predictions_array = probabilities_array.argmax(1)
#     print('class_predictions_array.shape: {}'.format(class_predictions_array.shape))

#     class_labels_array = onehot_labels_array.argmax(1)
    labels_array = labels_array
    accuracy = (class_predictions_array == labels_array).mean() * 100
#     accuracy = (class_predictions_array == class_labels_array).mean() * 100
    
    
    return accuracy, class_predictions_array, labels_array, probabilities_array

   
class RunningAverage():
    '''
    A class that maintains the running average of a quantity
    
    Usage example:
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    
    '''

    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total / float(self.steps)

    

def save_dict_to_json(d, json_path):
    """Saves dict of floats in josn file
    
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float)
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
    
    

def save_checkpoint(state, is_best, checkpoint):
    """Save model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves checkpoint + 'best.pth.tar'
    
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    
    else:
        print("Checkpoint Directory exists!")
    
    torch.save(state, filepath)
    
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
    


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. 
    If optimizer is provided, loads state_dict of optimizer assuming it is present in checkpoint.
    
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    
    return checkpoint
    
    
def write_model_info(model_state_dict, result_save_path, file_name):
    temp_file_name = os.path.join(result_save_path, file_name)
    
    auto_file = open(temp_file_name, 'w')
    total_elements = 0
    for name, tensor in model_state_dict.items():
        total_elements += torch.numel(tensor)
        auto_file.write('\t Layer {}: {} elements \n'.format(name, torch.numel(tensor)))

        #print('\t Layer {}: {} elements'.format(name, torch.numel(tensor)))
    auto_file.write('\n total elemets in this model state_dict: {}\n'.format(total_elements))
    #print('\n total elemets in this model state_dict: {}\n'.format(total_elements))
    auto_file.close()
    


