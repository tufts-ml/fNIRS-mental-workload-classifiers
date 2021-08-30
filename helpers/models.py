import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



#hz confirmed with keras implemenation of the official EEGNet: https://github.com/vlawhern/arl-eegmodels/blob/f3218571d1e7cfa828da0c697299467ea101fd39/EEGModels.py#L359


#assume using window size 200ts
#feature_size = 8
#timestep = 200
#F1 = 16
#D = 2
#F2 = D * F1
#output of each layer see HCI/NuripsDataSet2021/ExploreEEGNet_StepByStep.ipynb

#Conv2d with Constraint (https://github.com/braindecode/braindecode/blob/master/braindecode/models/eegnet.py)
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        
    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        
        return super(Conv2dWithConstraint, self).forward(x)
    


    

class EEGNet150(nn.Module):
    def __init__(self, feature_size=8, num_timesteps=150, num_classes=2, F1=4, D=2, F2=8, dropout=0.5):

        super(EEGNet150, self).__init__()

        #Temporal convolution
        self.firstConv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False), 
            #'same' padding: used by the author; 
            #kernel_size=(1,3):  author recommend kernel length be half of the sampling rate
            nn.BatchNorm2d(num_features=F1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            Conv2dWithConstraint(F1, F1 * D, kernel_size=(feature_size, 1), stride=(1, 1), groups=F1, bias=False), 
            #'valid' padding: used by the author;
            #kernel_size = (feature_size, 1): used by the author
            
            nn.BatchNorm2d(F1 * D, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #used by author
            nn.AvgPool2d(kernel_size=(1, 4)), #kernel_size=(1,4) used by author
            nn.Dropout(p=dropout) 
        )

        #depthwise convolution follow by pointwise convolution (pointwise convolution is just Conv2d with 1x1 kernel)
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=F1*D, bias=False),
            nn.Conv2d(F2, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
            nn.ELU(), #use by author
            nn.AvgPool2d(kernel_size=(1, 8)), #kernel_size=(1,8): used by author
            nn.Dropout(p=dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=32, out_features=num_classes, bias=True)
        ) #先不implement最后一层的kernel constraint， 只implement conv2d的constraint

    def forward(self, x):
        x = self.firstConv(x.unsqueeze(1).transpose(2,3))
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        # print(x.shape)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.classifier(x)
        normalized_probabilities= F.log_softmax(x, dim = 1)     

        return normalized_probabilities #for EEGNet and DeepConvNet, directly use nn.NLLLoss() as criterion
    



#the author used kernel_size=(1,3) stride=(1,3) for all the MaxPool2d layer. Here we use less agressive down-sampling, because our input chunk has only 200 timesteps

#we didn't implement the tied-loss as described by the author, because our goal is to predict each chunk, while the goal of the paper is to predict each trial from all the chunks of this trial.
    
class DeepConvNet150(nn.Module):
    def __init__(self, feature_size=8, num_timesteps=150, num_classes=2, dropout=0.5):
        super(DeepConvNet150, self).__init__()
        
    
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1,5), stride=1, padding=0, bias=True),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(feature_size, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.block2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1,5), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.block3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 5), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.block4 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 5), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=200, out_channels=num_classes, kernel_size=(1, 5), bias=True)
        )

        
    def forward(self, x):
        x = self.block1(x.unsqueeze(1).transpose(2,3))
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        x = x.squeeze(dim=2).squeeze(dim=2)
        print(x.shape)
        normalized_probabilities = F.log_softmax(x, dim = 1)     

        return normalized_probabilities #for EEGNet and DeepConvNet, directly use nn.NLLLoss() as criterion
    
    
