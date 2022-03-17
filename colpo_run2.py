import sys, os
sys.path.append(os.path.abspath('..'))
from colpo_train_evaluate import train_eval
from labeling_mechanisms import label_data
import numpy as np


data_folder= '/home/manalo/code/sar-pu-test/data/colpo_faster_rcnn'
dataset_folder= '/home/manalo/code/sar-pu-test/data/folds/'
model_name = 'resnet-50'
data_name = "colpo"

models = ['resnet-101','densenet-121']

for m in models:
    results_folder="/home/manalo/code/sar-pu-test/results/extended/"+m+"/cancer-unweighted-frozelastc/"
    pos_class = [2]
    for i in range(1):
        train_eval(
            data_folder,
            dataset_folder+str(i),
            results_folder+str(i), 
            model_name=m, 
            class_count=1,
            batch_size=64,
            seed=11,
            device_num='2,3',
            pos_class=pos_class
        )


    results_folder="/home/manalo/code/sar-pu-test/results/extended/"+m+"/cin23-unweighted-frozelastc/"
    pos_class = [1]
    for i in range(1):
        train_eval(
            data_folder,
            dataset_folder+str(i),
            results_folder+str(i), 
            model_name=m, 
            class_count=1,
            batch_size=64,
            seed=11,
            device_num='2,3',
            pos_class=pos_class
        )

    results_folder="/home/manalo/code/sar-pu-test/results/extended/"+m+"/cin1-unweighted-frozelastc/"
    pos_class = [0]
    for i in range(1):
        train_eval(
            data_folder,
            dataset_folder+str(i),
            results_folder+str(i), 
            model_name=m, 
            class_count=1,
            batch_size=64,
            seed=11,
            device_num='2,3',
            pos_class=pos_class
        )

    results_folder="/home/manalo/code/sar-pu-test/results/extended/"+m+"/cin23cancer-unweighted-frozelastc/"
    pos_class = [1,2]
    for i in range(1):
        train_eval(
            data_folder,
            dataset_folder+str(i),
            results_folder+str(i), 
            model_name=m, 
            class_count=1,
            batch_size=64,
            seed=11,
            device_num='2,3',
            pos_class=pos_class
        )


    results_folder="/home/manalo/code/sar-pu-test/results/extended/"+m+"/cincancer-unweighted-frozelastc/"
    pos_class = [0,1,2]
    for i in range(1):
        train_eval(
            data_folder,
            dataset_folder+str(i),
            results_folder+str(i), 
            model_name=m, 
            class_count=1,
            batch_size=64,
            seed=11,
            device_num='2,3',
            pos_class=pos_class
        )



