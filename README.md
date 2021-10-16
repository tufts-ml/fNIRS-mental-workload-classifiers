# fNIRS-mental-workload-classifiers
Code for training, evaluating, and visualizing performance of mental workload classification using fNIRS BCI sensors

For users who want to directly try our data on their own pipeline, we provide a simple demo of how to load the data and convert it to standard pytorch dataloader. Please see [DataLoader_Demo.ipynb](DataLoader_Demo.ipynb).

# Setup
### Download dataset
Please visit our website https://tufts-hci-lab.github.io/code_and_datasets/fNIRS2MW.html and download the data.
Extract the slide window data into data/slide_window_data folder.

### Install Anaconda
Follow the instructions here: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

### Create environment
conda env create -f environment.yml

# Running experiments

Code for runing each experiment in the paper are located in their own folders:

[SelectWindowSize](SelectWindowSize/): optimal window size experiments using Random Forest and Logistic Regression

[subject_specific_models](subject_specific_models/): subject-speicific models using DeepConvNet/EEGNet/Logistic Regression and Random Forest with the selected window size of 30sec

[generic_model](generic_models/): generic-models using DeepConvNet/EEGNet/Logistic Regression and Random Forest with the selected window size of 30sec. 3 scenarios of the generic pool size are experimented (64, 16, 4) 

[generic_finetuning_models](generic_finetuning_models/): finetuning the DeepConvNet and EEGNet from corresponding checkpoint of the 64-subject generic pool models with the selected window size of 30sec.

[domain_adaptation](domain_adaptation/): utilizing CORAL with Logistic Regression and Random Forest with the selected window size of 30sec. 1 scenario is experimented (generic pool size of 64 subjects and utilizing the target subject's full train set for domain adaptation) 

[subgroup_analysis](subgroup_analysis/): training a generic model solely from subjects of one subgroup's, and see how the performance generalize to othe subgroups with the selected window size of 30sec. Two scenario are experimented (training on White and training on Asian) (since these are the two majority groups in our dataset) 

The commands for reproducing results in the paper are provided in [runs](runs/) subfolders inside each experiment folders.

### Define the environment variable
```export YOUR_PATH="paths to this repo" ```
(e.g., '/home/usr/fNIRS-mental-workload-classifiers', then YOUR_PATH = '/home/usr')


### Demo
For example, if you want to train subject-specific DeepConvNet. Go to [runs](subject_specific_models/runs/window_size150) within the subject_sepcific_models folder
**(Please modify the paths in the scripts according to your actual paths)**

```
bash launch_DeepConvNet_150ts.sh run_here
```


# Analysing results

training curves, confusion matrix, checkpoints etc will be automatically saved in the specified directory in the after running the training commands.  

### Synthesizing hyperparameter search results

we also provide scripts for synthesizing hyperparameter search results [synthesizing_results](synthesizing_results/). For example, if you want to analyse results for subject-specific DeepConvNet, go to the [corresponding folder](synthesizing_results/subject_specific_models/binary/DeepConvNet) within the synthesizing_results/ folder. 

**(Please modify the paths in the scripts according to your actual paths)**

```
bash launch_DeepConvNet_150ts.sh run_here
```
This will generate a csv file called **hypersearch_summary.csv** for each subject. 

```
bash launch_synthesize_all_subjects.sh run_here
```

This will generate a csv file called **AllSubjects_summary.csv** for this experiment.
