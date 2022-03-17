import sys, os
sys.path.append(os.path.abspath('..'))
from colpo_train_evaluate import train_eval
import numpy as np


data_folder= '/home/manalo/code/sar-pu/data/colpo_faster_rcnn'
dataset_folder= '/home/manalo/code/sar-pu/data/folds/'
model_name = 'resnet-50'
data_name = "colpo"



models = ['resnet-50']

for m in models:
    results_folder="/home/manalo/code/sar-pu/results/"+m+"gradflowcheck/"
    for i in range(1):
        train_eval(
        data_folder,
        dataset_folder+str(i),
        results_folder+str(i), 
        model_name=m, 
        class_count=3,
        batch_size=64,
        seed=11,
        device_num='0,1,2,3',
        refit_classifier=True,
        device='cuda',
    )

#for m in models:
#    for w in (0.8, 0.6, 0.4, 0.2):
#        results_folder="/home/manalo/code/sar-pu/results/"+m+"revanillaweights/"
#        for i in range(1):
#            train_eval(
#                data_folder,
#                dataset_folder+str(i),
#                results_folder+str(i)+"_"+str(w), 
#                model_name=m, 
#                class_count=3,
#                batch_size=64,
#                seed=11,
#                device_num='0,1,2,3',
#                refit_classifier=True,
#                device='cuda',
#                prop_weight=w
#            )
#
#for m in models:
#    for w in (0.8, 0.6, 0.4, 0.2):
#        results_folder="/home/manalo/code/sar-pu/results/"+m+"revanillaweights/"
#        for i in range(2,3):
#            train_eval(
#                data_folder,
#                dataset_folder+str(i),
#                results_folder+str(i)+"_"+str(w), 
#                model_name=m, 
#                class_count=3,
#                batch_size=64,
#                seed=11,
#                device_num='0,1,2,3',
#                refit_classifier=True,
#                device='cuda',
#                prop_weight=w
#            )
#
#for m in models:
#    for w in (0.8, 0.6, 0.4, 0.2):
#        results_folder="/home/manalo/code/sar-pu/results/"+m+"revanillaweights/"
#        for i in range(3,4):
#            train_eval(
#                data_folder,
#                dataset_folder+str(i),
#                results_folder+str(i)+"_"+str(w), 
#                model_name=m, 
#                class_count=3,
#                batch_size=64,
#                seed=11,
#                device_num='0,1,2,3',
#                refit_classifier=True,
#                device='cuda',
#                prop_weight=w
#            )
#
#for m in models:
#    for w in (0.8, 0.6, 0.4, 0.2):
#        results_folder="/home/manalo/code/sar-pu/results/"+m+"revanillaweights/"
#        for i in range(4,5):
#            train_eval(
#                data_folder,
#                dataset_folder+str(i),
#                results_folder+str(i)+"_"+str(w), 
#                model_name=m, 
#                class_count=3,
#                batch_size=64,
#                seed=11,
#                device_num='0,1,2,3',
#                refit_classifier=True,
#                device='cuda',
#                prop_weight=w
#            )
#
#for m in models:
#    for w in (0.8, 0.6, 0.4, 0.2):
#        results_folder="/home/manalo/code/sar-pu/results/"+m+"revanillaweights/"
#        for i in range(1,2):
#            train_eval(
#                data_folder,
#                dataset_folder+str(i),
#                results_folder+str(i)+"_"+str(w), 
#                model_name=m, 
#                class_count=3,
#                batch_size=64,
#                seed=11,
#                device_num='0,1,2,3',
#                refit_classifier=True,
#                device='cuda',
#                prop_weight=w
#            )


#    results_folder="/home/manalo/code/sar-pu-test/results/extended/"+m+"/cin23-unweighted-frozelastc/"
#    pos_class = [1]
#    for i in range(1):
#        train_eval(
#            data_folder,
#            dataset_folder+str(i),
#            results_folder+str(i), 
#            model_name=m, 
#            class_count=1,
#            batch_size=64,
#            seed=11,
#            device_num='0,1',
#            pos_class=pos_class
#        )
#
#    results_folder="/home/manalo/code/sar-pu-test/results/extended/"+m+"/cin1-unweighted-frozelastc/"
#    pos_class = [0]
#    for i in range(1):
#        train_eval(
#            data_folder,
#            dataset_folder+str(i),
#            results_folder+str(i), 
#            model_name=m, 
#            class_count=1,
#            batch_size=64,
#            seed=11,
#            device_num='0,1',
#            pos_class=pos_class
#        )
#
#for m in models:
#    results_folder="/home/manalo/code/sar-pu-test/results/extended/"+m+"/cin23cancer-unweighted-frozelastc/"
#    pos_class = [1,2]
#    for i in range(1):
#        train_eval(
#            data_folder,
#            dataset_folder+str(i),
#            results_folder+str(i), 
#            model_name=m, 
#            class_count=1,
#            batch_size=64,
#            seed=11,
#            device_num='0,1',
#            pos_class=pos_class
#        )
#
#
#    results_folder="/home/manalo/code/sar-pu-test/results/extended/"+m+"/cincancer-unweighted-frozelastc/"
#    pos_class = [0,1,2]
#    for i in range(1):
#        train_eval(
#            data_folder,
#            dataset_folder+str(i),
#            results_folder+str(i), 
#            model_name=m, 
#            class_count=1,
#            batch_size=64,
#            seed=11,
#            device_num='0,1',
#            pos_class=pos_class
#        )
#

