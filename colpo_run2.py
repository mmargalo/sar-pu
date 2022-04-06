import sys, os
sys.path.append(os.path.abspath('..'))
import numpy as np
from main import main

data_folder= '/home/manalo/code/sar-pu/data/colpo_faster_rcnn'
dataset_folder= '/home/manalo/code/sar-pu/data/folds/'
model_name = 'resnet-50'
data_name = "colpo"



models = ['singlefc']
for m in models:
    results_folder="/home/manalo/code/sar-pu-git/results/"+m+"/"
    for i in range(1):
        main(
            data_folder, 
            dataset_folder+str(i), 
            results_folder+str(i), 
            model_name='resnetfeat-50', 
            c_model_name='singlefc-2048', 
            p_model_name='singlefc-2048', 
            class_count=3, 
            batch_size=64, 
            pos_class=None, 
            device='cuda', 
            seed=11, 
            device_num='0,1,2,3',
            outer_epochs=50, 
            inner_epochs=1)