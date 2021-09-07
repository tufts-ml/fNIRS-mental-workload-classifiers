import os
import numpy as np
import csv
import argparse
import pandas as pd

def main(experiment_dir, summary_save_dir):
    
    AllSubject_summary_filename = os.path.join(summary_save_dir, 'AllSubjects_summary.csv')
    
    with open(AllSubject_summary_filename, mode='w') as csv_file:
        
        fieldnames = ['subject_id', 'bucket', 'max_validation_accuracy', 'corresponding_test_accuracy', 'performance_string', 'experiment_folder']
        
        fileEmpty = os.stat(AllSubject_summary_filename).st_size == 0
        
        writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
        
        if fileEmpty:
            writer.writeheader()
        
#         subject_list = [1, 13, 14, 15, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 5, 51, 52, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 67, 68, 69, 7, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 91, 92, 93, 94, 95, 97]
        
        buckets = ['TestBucket1', 'TestBucket2', 'TestBucket3', 'TestBucket4', 'TestBucket5', 'TestBucket6', 'TestBucket7', 'TestBucket8', 'TestBucket9', 'TestBucket10', 'TestBucket11', 'TestBucket12', 'TestBucket13', 'TestBucket14', 'TestBucket15', 'TestBucket16', 'TestBucket17']
        
        
        for bucket in buckets:
            subject_this_bucket_list = os.listdir(os.path.join(experiment_dir, bucket))
        
            for subject_id in subject_this_bucket_list:
                this_subject_summary_csv_path = os.path.join(experiment_dir, bucket, str(subject_id), 'hypersearch_summary/hypersearch_summary.csv')

                this_subject_summary_df = pd.read_csv(this_subject_summary_csv_path)

                this_subject_selected_setting = this_subject_summary_df.sort_values(by=['validation_accuracy'], ascending=False).iloc[0]

                this_subject_dict = {}
                this_subject_max_validation_accuracy = this_subject_selected_setting.validation_accuracy
                this_subject_corresponding_test_accuracy = this_subject_selected_setting.test_accuracy
                this_subject_performance_string = this_subject_selected_setting.performance_string
                this_subject_experiment_folder = this_subject_selected_setting.experiment_folder

                this_subject_dict.update(subject_id=subject_id, bucket=bucket, max_validation_accuracy=this_subject_max_validation_accuracy, corresponding_test_accuracy=this_subject_corresponding_test_accuracy, performance_string=this_subject_performance_string, experiment_folder=this_subject_experiment_folder)

                writer.writerow(this_subject_dict)
            

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir')
    
    #parse args
    args = parser.parse_args()
    
    experiment_dir = args.experiment_dir
    summary_save_dir = experiment_dir + '_summary'
    
    if not os.path.exists(summary_save_dir):
        os.makedirs(summary_save_dir)
        
    main(experiment_dir, summary_save_dir)
    
    
    
    
    
        
        