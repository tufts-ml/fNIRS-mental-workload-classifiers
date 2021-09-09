import os
import numpy as np
import csv
import argparse

def extract_experiment_setting(experiment_name):
    '''
    extract the hyperparamter for LR: C
    '''
    print('Passed in experiment_name is {}'.format(experiment_name), flush = True)
    
    hyper_parameter_dict = {}
    
    #hyperparameter to extract
    MaxFeatures = experiment_name.split('MaxFeatures')[-1].split('_')[0]
    MinSamplesLeaf = experiment_name.split('MinSamplesLeaf')[-1]
    
    #record to dict
    hyper_parameter_dict['MaxFeatures'] = MaxFeatures
    hyper_parameter_dict['MinSamplesLeaf'] = MinSamplesLeaf
    
    #print values
    header = ' checking experiment '.center(100, '-')
    print(header)
    print('MaxFeatures: {}, MinSamplesLeaf: {}'.format(MaxFeatures, MinSamplesLeaf)) 
    
    print('\n')
    
    return hyper_parameter_dict

def extract_experiment_performance(experiment_dir, experiment_name):
    '''
    experiment_dir: experiments/FixedTrainValSplit_subject_specific_models/RandomForest/binary/window_size200/1
    experiment_name: MaxFeatures0.166_MinSamplesLeaf16
    '''
    
    performance_file_fullpath = os.path.join(experiment_dir, experiment_name, 'result_analysis/performance.txt')
    returned_file = None
    
    with open(performance_file_fullpath, 'r') as f: #only read mode, do not modify
        returned_file = f.read()
        
        validation_accuracy = round(float(returned_file.split('highest validation accuracy: ')[1].split('\n')[0]), 3)
        test_accuracy = returned_file.split('corresponding test accuracy: ')[1].split('\n')[0]
        
        print('validation_accuracy: {}'.format(validation_accuracy))
        print('test_accuracy: {}'.format(test_accuracy))
        
    return returned_file, validation_accuracy, test_accuracy


def main(experiment_dir, summary_save_dir):
    
    experiments = os.listdir(experiment_dir)
    incomplete_experiment_writer = open(os.path.join(summary_save_dir, 'incomplete_experiment_list.txt'), 'w')
    summary_filename = os.path.join(summary_save_dir, 'hypersearch_summary.csv')
    
    with open(summary_filename, mode='w') as csv_file:
        
        fieldnames = ['validation_accuracy', 'test_accuracy', 'MaxFeatures', 'MinSamplesLeaf', 'performance_string', 'experiment_folder', 'status']
        fileEmpty = os.stat(summary_filename).st_size==0
        
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if fileEmpty:
            writer.writeheader()
        
        for experiment_name in experiments:
            if experiment_name !='hypersearch_summary':
                experiment_folder = os.path.join(experiment_dir, experiment_name)
                
                experiment_summary = extract_experiment_setting(experiment_name)
                
                try:
                    returned_file, validation_accuracy, test_accuracy = extract_experiment_performance(experiment_dir, experiment_name)
                    print('Able to extract performance', flush = True)
                    
                    experiment_summary.update(validation_accuracy=validation_accuracy, test_accuracy=test_accuracy, performance_string=returned_file, experiment_folder=experiment_folder, status='Completed')
                    print('Able to update experiment_summary\n\n')
                
                except:
                    print(' NOT ABLE TO PROCESS {} \n\n'.format(experiment_dir + '/' + experiment_name).center(100, '-'), flush=True)
                    
                    incomplete_experiment_writer.write(f"{experiment_name}\n\n")
                    experiment_summary.update(validation_accuracy='NA', test_accuracy='NA', performance_string='NA', experiment_folder=experiment_folder, status='Incompleted')
                    
                writer.writerow(experiment_summary)
            
        incomplete_experiment_writer.close()
        
        

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='synthesizing hyperparameter search results')
    parser.add_argument('--experiment_dir')
    
    #parse args
    args = parser.parse_args()
    
    experiment_dir = args.experiment_dir
    assert os.path.exists(experiment_dir),'The passed in experiment_dir {} does not exist'.format(experiment_dir)
    
    summary_save_dir = os.path.join(experiment_dir, 'hypersearch_summary')
    
    if not os.path.exists(summary_save_dir):
        os.makedirs(summary_save_dir)
    
    main(experiment_dir, summary_save_dir)
    
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    