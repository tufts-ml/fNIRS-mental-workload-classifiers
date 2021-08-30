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
    
#     if SelectWindowSize_testset:
#         if classification_task == 'four_class':
#             assert num_data == 608
#         elif classification_task == 'binary':
#             assert num_data == 304
#         else:
#             raise NameError('not supported classification setting')
#     else:
#         if classification_task == 'four_class':
#             assert num_data == 1112
#         elif classification_task == 'binary':
#             assert num_data == 556
#         else:
#             raise NameError('not supported classification setting')

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
    
    
def save_training_curves(figure_name, result_save_subject_trainingcurvedir, epoch_train_loss, epoch_validation_accuracy = None):
    
    fig = plt.figure(figsize=(15, 8))
    
    ax_1 = fig.add_subplot(1,2,1)
    ax_1.plot(range(len(epoch_train_loss)), epoch_train_loss, label='epoch_train_loss')
    
    if epoch_validation_accuracy is not None:
        ax_2 = fig.add_subplot(1,2,2, sharex = ax_1)
        ax_2.plot(range(len(epoch_validation_accuracy)), epoch_validation_accuracy, label='epoch_validation_accuracy')
        ax_2.legend()
    
    ax_1.legend()
        
    figure_save_path = os.path.join(result_save_subject_trainingcurvedir, figure_name)
    plt.savefig(figure_save_path)
    plt.close()
    

def ensemble_and_extract_performance(model_state_dict, result_save_subject_predictionsdir, result_save_subject_resultanalysisdir, save_dict_name = 'result_save_dict.pkl'):
    '''
    used for extracting performance for phase3 model
    '''
    
    #create file writer
    file_writer = open(os.path.join(result_save_subject_resultanalysisdir, 'performance.txt'), 'w')
    
    #load saved test predictions
    CV_predictions_dict = load_pickle(result_save_subject_predictionsdir, save_dict_name)
    
    assert np.array_equal(CV_predictions_dict['fold0_bestepoch_test_class_labels'], CV_predictions_dict['fold1_bestepoch_test_class_labels']) and np.array_equal(CV_predictions_dict['fold1_bestepoch_test_class_labels'], CV_predictions_dict['fold2_bestepoch_test_class_labels']) and np.array_equal(CV_predictions_dict['fold2_bestepoch_test_class_labels'], CV_predictions_dict['fold3_bestepoch_test_class_labels']) and np.array_equal(CV_predictions_dict['fold3_bestepoch_test_class_labels'], CV_predictions_dict['fold4_bestepoch_test_class_labels']), 'test set should not shuffle'
    
    
    #perform ensemble
    predictions_to_ensemble = [CV_predictions_dict['fold{}_bestepoch_test_logits'.format(i)] for i in range(5)]
    true_labels = CV_predictions_dict['fold0_bestepoch_test_class_labels']
    
    bagging_accuracy, bagging_class_predictions, bagging_logits = simple_bagging(predictions_to_ensemble, true_labels)
    
    average_cv_validation_accuracy = np.mean(np.array([CV_predictions_dict['fold{}_bestepoch_val_accuracy'.format(i)] for i in range(5)]))
    #write performance to file
    file_writer.write('Average cv validation accuracy: {}\n\n'.format(average_cv_validation_accuracy))

    
    file_writer.write('Test accuracy:\n')
    file_writer.write('fold0: {}\n'.format(CV_predictions_dict['fold0_bestepoch_test_accuracy']))
    file_writer.write('fold1: {}\n'.format(CV_predictions_dict['fold1_bestepoch_test_accuracy']))
    file_writer.write('fold2: {}\n'.format(CV_predictions_dict['fold2_bestepoch_test_accuracy']))
    file_writer.write('fold3: {}\n'.format(CV_predictions_dict['fold3_bestepoch_test_accuracy']))
    file_writer.write('fold4: {}\n\n'.format(CV_predictions_dict['fold4_bestepoch_test_accuracy']))
    file_writer.write('Ensemble_5folds: {}\n\n'.format(bagging_accuracy))
    
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
        
        

    return bagging_accuracy, bagging_class_predictions, bagging_logits

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

    
def simple_bagging(predictions_to_ensemble, true_labels):
    
    assert len(predictions_to_ensemble) == 5
    
    running_predictions = 0
    for predictions in predictions_to_ensemble:
        print('predictions.shape: {}'.format(predictions.shape))
        running_predictions += predictions
    
    ensembled_class_predictions = running_predictions.argmax(1)
    ensembled_logits = running_predictions/len(predictions_to_ensemble)
    ensembled_accuracy = (ensembled_class_predictions == true_labels).mean() * 100
    
    return ensembled_accuracy, ensembled_class_predictions, ensembled_logits
    


#for pytorch, need to define soft cross entropy (no built-in supported)
#https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501

def soft_cross_entropy(logits, soft_target):
    '''
    soft_target: e.g., [0.1, 0.2, 0.6, 0.1]
    '''
#     print('logits: {}'.format(type(logits)))
#     print('soft_target: {}'.format(type(soft_target)))
    logprobs = F.log_softmax(logits, dim=1)
    
    soft_cross_entropy_loss = -(soft_target * logprobs).sum()/logits.shape[0]
#     print('soft_cross_entropy_loss: {}'.format(type(soft_cross_entropy_loss)))
   
    return soft_cross_entropy_loss


def convert_numpyarray_to_onehot(input_array, n_values=4):
    
#     n_values = 4 #0back, 1back, 2back, 3back
    onehot_encoding = np.eye(n_values)[input_array]
    
    print('Inside convert_numpyarray_to_onehot, onehot_encoding shape is {}'.format(onehot_encoding.shape), flush=True)
    
    return onehot_encoding
        

def MixUp_expansion(prior_sub_feature_array, prior_sub_label_array, alpha = 0.75, expand=2):
    
    '''
    Mixing strategy1: mixing same chunk of different person to create synthetic person
                      randomly choose two person, sample lambda from beta distribution, use the same beta for each chunk
    '''
    
    assert len(prior_sub_feature_array) == len(prior_sub_label_array)
    assert isinstance(prior_sub_feature_array, np.ndarray), 'input_images is not numpy array'
    assert isinstance(prior_sub_label_array, np.ndarray), 'input_labels is not numpy array'
    
    
#     expanded_sub_feature_array = np.array(prior_sub_feature_array).copy()
#     expanded_sub_label_array = np.array(prior_sub_label_array).copy()
   
    expanded_sub_feature_array = None
    expanded_sub_label_array = None
    
    num_sub = len(prior_sub_feature_array)
    
    for i in range(expand):
        lam = np.random.beta(alpha, alpha, (num_sub, 1, 1, 1))
        lam = np.maximum(lam, (1 - lam)) #ensure the created samples is closer to the first sample
        
        permutation_indices = np.random.permutation(num_sub)
        
        #linear interpolation of features
        synthetic_sub_feature_array = prior_sub_feature_array * lam + prior_sub_feature_array[permutation_indices] * (1 - lam)

        #linear interpolation of labels
        synthetic_sub_label_array = prior_sub_label_array * lam[:, :, :, 0] + prior_sub_label_array[permutation_indices] * (1 - lam[:, :, :, 0])  
    
        if expanded_sub_feature_array is None:
            expanded_sub_feature_array = synthetic_sub_feature_array
            expanded_sub_label_array = synthetic_sub_label_array
        else:     
            expanded_sub_feature_array = np.concatenate((expanded_sub_feature_array, synthetic_sub_feature_array))
            expanded_sub_label_array = np.concatenate((expanded_sub_label_array, synthetic_sub_label_array))
    
    return expanded_sub_feature_array, expanded_sub_label_array
    
    
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


#Aug13 
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

    

def set_logger(log_path):
    """Set the logger to log info in terminal and file 'log_path'.
    
    In general, it is useful to have a logger so that every output to the terminal is saved in a 
    permanent file. Here we save it to 'model_dir/train.log'
    
    Example:
    '''
    logging.info("Start training...")
    '''
    
    Args:
        log_path: (string) where to log
    """
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)    
    
    

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
    
    

#June15: writing model information:
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
    


