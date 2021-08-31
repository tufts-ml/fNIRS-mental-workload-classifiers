#data set class

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob


class brain_dataset(Dataset):

    def __init__(self, instance_list, label_list):
        self.instance_list = instance_list
        self.instance_label = label_list
           
    def __getitem__(self, index):
        return self.instance_list[index], self.instance_label[index]
    
    def __len__(self):
        return len(self.instance_list)
    
    def __get_instance_label__(self):
        return self.instance_label
    
    def __get_instance_list__(self):
        return np.array(self.instance_list).shape
    


def read_subject_csv_binary(path, select_feature_columns = ['AB_I_O', 'AB_PHI_O', 'AB_I_DO', 'AB_PHI_DO', 'CD_I_O', 'CD_PHI_O',
       'CD_I_DO', 'CD_PHI_DO'], num_chunk_this_window_size = 2224):
    
    '''
    For binary classification: 0 vs 2
    '''
    
    instance_list = []
    instance_label = []
    
    # each subject csv file contain 2224 chunks (for window size 10 stride 3)
    subject_df = pd.read_csv(path) 
    assert np.max(subject_df.chunk.values) + 1 == num_chunk_this_window_size, '{} does not have {} chunks'.format(path, num_chunk_this_window_size) 
    
    subject_df = subject_df[select_feature_columns + ['chunk'] + ['label']] 

    #chunk id: 0 to 2223 (for window size 10 stride 3)
    for i in range(0, num_chunk_this_window_size):
        print('current chunk: {}'.format(i))
        chunk_matrix = subject_df.iloc[:,:-2].loc[subject_df['chunk'] == i].values
        label_for_this_segment = subject_df[subject_df['chunk'] == i].label.values[0]
        
        if label_for_this_segment == 0:
            print('label_for_this_segment is {}'.format(label_for_this_segment), flush = True)
            instance_list.append(chunk_matrix)        
            instance_label.append(label_for_this_segment)
        elif label_for_this_segment == 2:
            print('label_for_this_segment is {}, map to class1'.format(label_for_this_segment), flush = True)
            instance_list.append(chunk_matrix)        
            instance_label.append(int(1))

#
            
    instance_list = np.array(instance_list, dtype=np.float32) 
    instance_label = np.array(instance_label, dtype=np.int64)
    
#     print('Inside brain_data, this subject data size: {}'.format(instance_label.shape[0]), flush = True)
#     assert instance_label.shape[0] == 1112
    
    
    return instance_list, instance_label



def read_subject_csv_binary_SelectWindowSize(path, select_feature_columns = ['AB_I_O', 'AB_PHI_O', 'AB_I_DO', 'AB_PHI_DO', 'CD_I_O', 'CD_PHI_O',
       'CD_I_DO', 'CD_PHI_DO']):
    
    '''
    For binary classification: 0 vs 2
    '''
    
    instance_list = []
    instance_label = []
    
    # each subject csv file contain 608 chunks 
    subject_df = pd.read_csv(path) 
    assert np.max(subject_df.chunk.values) + 1 == 608, '{} SelectWindowSize testset does not have 608 chunks'.format(path) 
    
    subject_df = subject_df[select_feature_columns + ['chunk'] + ['label']] 

    #chunk id: 0 to 608
    for i in range(0, 608):
        chunk_matrix = subject_df.iloc[:,:-2].loc[subject_df['chunk'] == i].values
        
        assert len(list(set(subject_df[subject_df['chunk'] == i].label.values))) == 1, 'each chunk has only 1 label'
        label_for_this_segment = subject_df[subject_df['chunk'] == i].label.values[0]
        
        if label_for_this_segment == 0:
            print('label_for_this_segment is {}'.format(label_for_this_segment), flush = True)
            instance_list.append(chunk_matrix)        
            instance_label.append(label_for_this_segment)
        elif label_for_this_segment == 2:
            print('label_for_this_segment is {}, map to class1'.format(label_for_this_segment), flush = True)
            instance_list.append(chunk_matrix)        
            instance_label.append(int(1))

#
            
    instance_list = np.array(instance_list, dtype=np.float32) 
    instance_label = np.array(instance_label, dtype=np.int64)
    
#     print('Inside brain_data, this subject SelectWindowSize testset size: {}'.format(instance_label.shape[0]), flush = True)
#     assert instance_label.shape[0] == 304
    
    
    return instance_list, instance_label


