from email.policy import strict
from os.path import join
from tensorboardX import SummaryWriter
import dill as pickle
import torch
from colpo_score import *
from colpo_sar_pu_em import train_sar_em, test_sar_em
from colpo_loader import setup_loader
from setup_multirun import nondeterministic
from model.model_factory import get_model
import os


def train_eval(data_folder, dataset_folder, results_folder, model_name, class_count=3, batch_size=64, pos_class=None, device='cuda', seed=None, device_num="0", test_only=False, snapshot=None, refit_classifier=True, prop_weight=1.):
    """
        Train and evaluate using SAR-PU
        :param data_folder: folder with images
        :param dataset_folder: folder with set divisions and data labels 
        :param results_folder: folder to store results
        :param class_count: number of classes, 1 for binary
        :param batch_size: batch size
        :param pos_class: list of positive classes based on label index  
        :param device: cpu or cuda
        :param seed: seed init
        :param device_num: gpu to use
        :return: nothing
    """

    # for reproducibility - set seed
    if seed: nondeterministic(seed)
    # set GPU to use
    if device_num: os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)

    test_dataloader = setup_loader(data_folder, join(dataset_folder, "test.txt"), class_count, batch_size = batch_size, train=False, seed=seed, pos_class=pos_class)
    
    model_name = model_name.split('-')
    model = get_model(model_name[0], model_name[1], class_count)

    model = model.to(device)
    
    #-----FOR FASTER R-CNN WEIGHTS----------------------
    # weights = torch.load("./weights/frcnn.pth")
    # for key in list(weights["state_dict"].keys()):
    #     if "backbone.body." in key:
    #         weights["state_dict"][key.replace("backbone.body.", "")] = weights["state_dict"].pop(key)
    #     else:
    #         weights["state_dict"].pop(key)
    # model.load_state_dict(weights["state_dict"], strict=False)
    #---------------------------------------------------

    #-----FOR SIMCLR WEIGHTS----------------------
    # weights = torch.load("./weights/simclr.pt")
    # for key in list(weights.keys()):
    #     if "encoder." in key:
    #         weights[key.replace("encoder.", "")] = weights.pop(key)
    #     else:
    #         weights.pop(key)
    # model.load_state_dict(weights, strict=False)
    #--------------------------------------------------
    
    if device == 'cuda' and torch.cuda.device_count() > 1 and len(device_num)>1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(os.path.join(dataset_folder, "./vanilla.pt")), strict=False) # has module, parallel

    if test_only:
        # load weights
        snapshot = snapshot if snapshot else join(results_folder, "BEST_checkpoint.pt")
        # model.load_state_dict(torch.load(snapshot))
    else:
        # Tensorboard writers
        train_writer = SummaryWriter(log_dir=join(results_folder, 'train'))
        val_writer = SummaryWriter(log_dir=join(results_folder, 'val'))
        
         # set up data loaders
        train_dataloader = setup_loader(data_folder, join(dataset_folder, "train.txt"), class_count, batch_size = batch_size, train=True, seed=seed, pos_class=pos_class)
        val_dataloader = setup_loader(data_folder, join(dataset_folder, "val.txt"), class_count, batch_size = batch_size, train=False, seed=seed, pos_class=pos_class)

        model = train_sar_em(model, device, train_dataloader, val_dataloader, train_writer=train_writer, val_writer=val_writer, results_folder=results_folder, refit_classifier=refit_classifier, prop_weight=prop_weight)

    #model.eval()
#
    #train_dataloader = setup_loader(data_folder, join(dataset_folder, "train.txt"), train=False, seed=seed, pos_class=pos_class)
    #output_test_f, output_test_e = test_sar_em(model, device, test_dataloader)
    #output_train_f, output_train_e = test_sar_em(model, device, train_dataloader)
#
    #train_labels = np.array(train_dataloader.dataset.get_labels(), dtype='float64')
    #test_labels = np.array(test_dataloader.dataset.get_labels(), dtype='float64')
#
    #results_test = evaluate_all(test_labels, output_test_f, output_test_e)
    #results_train = evaluate_all(train_labels, output_train_f, output_train_e)
#
    #results = {
    #    **{"train_" + k: v for k, v in results_train.items()},
    #    **{"test_" + k: v for k, v in results_test.items()}
    #}
#
    #out_results = join(results_folder, 'results.csv')
    #with open(out_results, "w+") as out_results_file:
    #    out_results_file.write("\n".join([k+"\t"+str(v) for k,v in sorted(results.items())]))

