from email.policy import strict
from os.path import join
from tensorboardX import SummaryWriter
import torch
from colpo_score import *
from colpo_sar_pu_em import train_val
from colpo_loader import setup_loader
from setup_multirun import nondeterministic
from model.model_factory import get_model
from loss_factory import get_loss
import os
from colpo_c_only import train_c_only

def main(data_folder, dataset_folder, results_folder, 
        model_name, c_model_name, p_model_name, 
        class_count=3, batch_size=64, pos_class=None, 
        device='cuda', seed=None, device_num="0",
        outer_epochs=50, inner_epochs=1, pretrain_weights=None,
        c_only=False):
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

    # set up data loaders
    train_loader = setup_loader(data_folder, join(dataset_folder, "train.txt"), 
                        class_count, batch_size = batch_size, train=True, 
                        seed=seed, pos_class=pos_class)
    val_loader = setup_loader(data_folder, join(dataset_folder, "val.txt"), 
                        class_count, batch_size = batch_size, train=False, 
                        seed=seed, pos_class=pos_class)
    # model setup
    model_name = model_name.split('-')
    backbone = get_model(model_name[0], model_name[1], class_count)
    backbone = backbone.to(device)

    c_model_name = c_model_name.split('-')
    c_model = get_model(c_model_name[0], c_model_name[1], class_count)
    c_model = c_model.to(device)

    p_model_name = p_model_name.split('-')
    p_model = get_model(p_model_name[0], p_model_name[1], class_count)
    p_model = p_model.to(device)

    if device == 'cuda':
        backbone = torch.nn.DataParallel(backbone)
        c_model = torch.nn.DataParallel(c_model)
        p_model = torch.nn.DataParallel(p_model)

    if pretrain_weights is not None:
         # vanilla has module, parallel
        backbone.load_state_dict(torch.load(os.path.join(dataset_folder, pretrain_weights+".pt")), strict=False)
        print("Loaded", pretrain_weights, "...")

    # optimizer setup
    optim = torch.optim.SGD(backbone.parameters(), lr=0.01)
    c_optim = torch.optim.SGD(c_model.parameters(), lr=0.01)
    p_optim = torch.optim.SGD(p_model.parameters(), lr=0.01)
    #optim = torch.optim.Adam(backbone.parameters(), lr=0.01)
    #c_optim = torch.optim.Adam(c_model.parameters(), lr=0.01)
    #p_optim = torch.optim.Adam(p_model.parameters(), lr=0.01)

    scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=0.0001, max_lr=0.01, step_size_up=52)
    c_scheduler = torch.optim.lr_scheduler.CyclicLR(c_optim, base_lr=0.00001, max_lr=0.01, step_size_up=52)
    p_scheduler = torch.optim.lr_scheduler.CyclicLR(p_optim, base_lr=0.00001, max_lr=0.01, step_size_up=52)

    # criterion setup
    criterion = get_loss("loglikelihood")
    c_criterion = get_loss("bcelogits")
    p_criterion = get_loss("multimargin")

    # Tensorboard writers
    train_writer = SummaryWriter(log_dir=join(results_folder, 'train'))
    val_writer = SummaryWriter(log_dir=join(results_folder, 'val'))

    if c_only:
        bb_weights = os.path.join(results_folder, "BEST_bb.pt")
        c_weights = os.path.join(results_folder, "BEST_c.pt")
        p_weights = os.path.join(results_folder, "BEST_p.pt")

        backbone.load_state_dict(torch.load(bb_weights),strict=True)
        c_model.load_state_dict(torch.load(c_weights),strict=True)
        p_model.load_state_dict(torch.load(p_weights),strict=True)

        c_scheduler = torch.optim.lr_scheduler.CyclicLR(c_optim, base_lr=0.001, max_lr=0.1, step_size_up=52)


        train_c_only(device, results_folder,
            backbone, c_model, p_model,
            train_loader, val_loader,
            criterion, c_optim, c_scheduler,
            train_writer, val_writer, 
            epochs=int(outer_epochs/2))
    else:

        train_val(results_folder, outer_epochs, inner_epochs,
                train_loader, val_loader, 
                train_writer, val_writer, 
                backbone, c_model, p_model, 
                criterion, c_criterion, p_criterion,
                optim, c_optim, p_optim,
                scheduler, c_scheduler, p_scheduler)

