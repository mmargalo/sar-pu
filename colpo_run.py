import sys, os
sys.path.append(os.path.abspath('..'))
import numpy as np
from main import main

data_folder= '/home/manalo/code/sar-pu/data/colpo_faster_rcnn'
#dataset_folder= '/home/manalo/code/sar-pu/data/folds/'
dataset_folder= '/home/manalo/code/sar-pu-git/data/'
model_name = 'resnet-50'
data_name = "colpo"



models = ['csra-batch128-lr01-cycliclr-ccstep40max01']
for m in models:
    results_folder="/home/manalo/code/sar-pu-git/results/attention/"+m+"/"
    for i in range(1):
        main(
            data_folder, 
            dataset_folder+str(i), 
            results_folder+str(i), 
            model_name='resnetfeat-49', 
            c_model_name='csraresnet-50', 
            p_model_name='csraresnet-50', 
            class_count=3, 
            batch_size=128, 
            pos_class=None, 
            device='cuda', 
            seed=11, 
            device_num='0',
            outer_epochs=100, 
            inner_epochs=1,
            c_only=False)

