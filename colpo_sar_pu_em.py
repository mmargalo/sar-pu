import numpy as np
import time
from pu_models import *
import torch
import torch.optim as optim
from tqdm import tqdm
from numpy import errstate, isneginf, inf
import math
import yaml
from copy import deepcopy
from os.path import join
from _utils import plot_grad_flow

def train_val(results_folder, outer_epochs, inner_epochs,
            train_loader, val_loader, 
            train_writer, val_writer, 
            backbone, c_model, p_model, 
            criterion, c_criterion, p_criterion,
            optim, c_optim, p_optim,
            scheduler, c_scheduler, p_scheduler):

    # for debugging
    torch.autograd.set_detect_anomaly(True)
    
    best_ll = 999.
    y1_train, y1_val = None, None
    for epoch in range(outer_epochs):
        print("EPOCH: " + str(epoch))
        _, y1_train = backbone_loop(epoch, train_loader, train_writer, results_folder,
                    backbone, c_model, p_model,
                    criterion, c_criterion, p_criterion,
                    optim, c_optim, p_optim, 
                    scheduler=scheduler, c_scheduler=c_scheduler, p_scheduler=p_scheduler,
                    inner_epochs=inner_epochs, lists=y1_train)
        ll, y1_val = backbone_loop(epoch, val_loader, val_writer, results_folder,
                    backbone, c_model, p_model, 
                    criterion, c_criterion, p_criterion,
                    scheduler=scheduler, c_scheduler=c_scheduler, p_scheduler=p_scheduler,
                    lists=y1_val)

        if ll < best_ll:
            save(backbone, c_model, p_model, results_folder, best=True)
            print("BEST LL: " + str(ll))
            best_ll = ll

        save(backbone, c_model, p_model, results_folder)


def backbone_loop(outer_epoch, dataloader, writer, results_folder,
                backbone, c_model, p_model, 
                criterion, c_criterion, p_criterion,
                optim=None, c_optim=None, p_optim=None,
                scheduler=None, c_scheduler=None, p_scheduler=None, 
                inner_epochs=1, device='cuda', lists=None):

    is_train = False if optim is None else True
    y1_post_list = lists
    toggle_downstream(c_model, p_model, True)
    y1_post_list = downstream_loop(outer_epoch, inner_epochs, writer, 
                    results_folder, dataloader,
                    backbone, c_model, p_model, 
                    c_criterion, p_criterion, 
                    c_optim, p_optim,
                    c_scheduler, p_scheduler, lists=y1_post_list)
    toggle_downstream(c_model, p_model, False)
    toggle_backbone(backbone, True)
    backbone.train() if is_train else backbone.eval()

    loss_total = 0.
    per_class = np.array([0.,0.,0.])
    criterion = criterion()
    for n_iter, (idx, x, labels) in enumerate(tqdm(dataloader)):
        x, labels = x.float().to(device), labels.float().to(device)
        
        with torch.set_grad_enabled(is_train):
            features = backbone(x)

            exp_prior_y1 = c_model(features)
            exp_propensity = p_model(features)

            if is_train: optim.zero_grad()
            loss = criterion(exp_prior_y1, exp_propensity, labels, weighted=True)
            loss, loss_total, per_class = update_loss(loss, loss_total, per_class)
            if is_train:
                loss.backward()
                plot_grad_flow(backbone, name="backbone", dest=results_folder)
                optim.step()

        # cycliclr
        if is_train:
            writer.add_scalar('lr', optim.param_groups[0]['lr'], outer_epoch * len(dataloader) + n_iter)
            scheduler.step()

    ll = loss_total/len(dataloader)

    #if is_train: 
    #    writer.add_scalar('lr', optim.param_groups[0]['lr'], outer_epoch)
    #else:
    #    # scheduler.step(ll)
    #    scheduler.step()
    
    writer.add_scalar('log_likelihood', ll, outer_epoch)
    for l_i, l in enumerate(per_class):
        writer.add_scalar('log_likelihood_'+str(l_i), l/len(dataloader), outer_epoch)

    return ll, y1_post_list


def downstream_loop(outer_epoch, epochs, writer, 
                    results_folder, dataloader, 
                    backbone, c_model, p_model, 
                    c_criterion, p_criterion, 
                    c_optim, p_optim,
                    c_scheduler, p_scheduler, device='cuda',
                    lists=None):
                    
    is_train = False if c_optim is None and p_optim is None else True

    c_model.train() if is_train else c_model.eval()
    p_model.train() if is_train else p_model.eval()

    if lists is None:
        # for epoch 0
        size = len(dataloader.dataset)
        y1_post_list = [None] * size
    else:
        y1_post_list = lists

    for epoch in range(epochs):
        c_loss_total = 0.
        p_loss_total = 0.
        per_c_loss = np.array([0.,0.,0.])
        per_p_loss = np.array([0.,0.,0.])

        for n_iter, (idx, x, s) in enumerate(tqdm(dataloader)):
            x, s = x.float().to(device), s.float().to(device)
            c_model.train() if is_train else c_model.eval()
            p_model.train() if is_train else p_model.eval()
            
            with torch.set_grad_enabled(is_train):
                toggle_backbone(backbone, False)
                features = backbone(x)

                exp_prior_y1 = c_model(features.detach())
                exp_propensity = p_model(features.detach())
                
                if outer_epoch == 0:
                    # INITIALIZE –do an entire pass on the whole training dataset
                    proportion_labeled = torch.sum(s, 0)/len(s)
                    # invert proportions as weights for labeled and unlabeled
                    c_exp = torch.nn.Sigmoid()(exp_prior_y1).view_as(s).detach()
                    p_weights = s + (1 - s) * c_exp
                    c_weights = s * (1 - proportion_labeled) + (1 - s) * proportion_labeled
                    #p_criterion2 = p_criterion(weight=p_weights, reduction='none')
                    p_criterion2 = p_criterion()
                    c_criterion2 = c_criterion(weight=c_weights, reduction='none')
                    if is_train:
                        c_optim.zero_grad()
                        p_optim.zero_grad()
                    p_loss = p_criterion2(exp_propensity, s, weights=p_weights)
                    c_loss = c_criterion2(exp_prior_y1, s)

                else:
                    exp_post_y1 = get_list(idx, y1_post_list).to(device)
                    #exp_post_y1 = expectation_y(exp_prior_y1, exp_propensity, s) 
                    p_criterion2 = p_criterion()
                    #p_criterion2 = p_criterion(weight=exp_post_y1.detach(), reduction='none')
                    c_criterion2 = c_criterion(reduction='none')
                    if is_train:
                        c_optim.zero_grad()
                        p_optim.zero_grad()
                    p_loss = p_criterion2(exp_propensity, s, weights=exp_post_y1.detach())
                    c_loss = c_criterion2(exp_prior_y1, exp_post_y1)
                
                p_loss, p_loss_total, per_p_loss = update_loss(p_loss, p_loss_total, per_p_loss)
                c_loss, c_loss_total, per_c_loss = update_loss(c_loss, c_loss_total, per_c_loss)

                if is_train:
                    p_loss.backward(retain_graph=True)
                    plot_grad_flow(p_model, name="propensity", dest=results_folder)
                    c_loss.backward()
                    plot_grad_flow(c_model, name="classifier", dest=results_folder)
                    p_optim.step()
                    c_optim.step()
                
                
                with torch.no_grad():
                    c_model.eval()
                    p_model.eval()
        
                    exp_prior_y1 = c_model(features)
                    exp_propensity = p_model(features)
                    exp_post_y1 = expectation_y(exp_prior_y1, exp_propensity, s)
                    exp_post_y1 = torch.nan_to_num(exp_post_y1, posinf=1.0)
                    y1_post_list = set_list(idx, exp_post_y1, y1_post_list)

                if is_train:
                    # cyclic lr
                    writer.add_scalar('lr_classifier', c_optim.param_groups[0]['lr'], outer_epoch* len(dataloader) + n_iter)
                    writer.add_scalar('lr_propensity', p_optim.param_groups[0]['lr'], outer_epoch* len(dataloader) + n_iter)

                    c_scheduler.step()
                    p_scheduler.step()
                     
        #if is_train:
        #    writer.add_scalar('lr_classifier', c_optim.param_groups[0]['lr'], outer_epoch*(epoch+1))
        #    writer.add_scalar('lr_propensity', p_optim.param_groups[0]['lr'], outer_epoch*(epoch+1))
        #else:
        #    # c_scheduler.step(c_loss_total/len(dataloader))
        #    # p_scheduler.step(p_loss_total/len(dataloader))

        writer.add_scalar('loss_classifier', c_loss_total/len(dataloader), outer_epoch*(epoch+1))
        writer.add_scalar('loss_propensity', p_loss_total/len(dataloader), outer_epoch*(epoch+1))
        for c_i, c in enumerate(per_c_loss):
            writer.add_scalar('loss_classifier_'+str(c_i), c/len(dataloader), outer_epoch*(epoch+1))
        for p_i, p in enumerate(per_p_loss):
            writer.add_scalar('loss_propensity_'+str(p_i), p/len(dataloader), outer_epoch*(epoch+1))

    return y1_post_list

def get_list(idx, list):
    out = []
    for i in idx:
        i = i.item()
        out.append(list[i])
    return torch.FloatTensor(out)

def set_list(idx, items, list):
    
    for i, item in zip(idx, items.detach().clone()):
        detached = [x.item() for x in item]
        list[i.item()] = detached
    return list

def update_loss(loss, total, per_class):
    # updates total and per class losses, returns them with the avg loss
    total = total + loss.mean().item()
    loss_per = [l.item() for l in torch.mean(loss, 0)]
    per_class = per_class + np.array(loss_per)
    return torch.mean(loss), total, per_class


def expectation_y(exp_f, exp_e, s):
    sigmoid = torch.nn.Sigmoid()
    f = sigmoid(exp_f)
    e = sigmoid(exp_e)
    return s+(1-s)*(f*(1-e))/(1-f*e)


def save(model, c_model, p_model, dest, name="checkpt", best=False):
    torch.save(model.state_dict(), dest + "/"+name+"_bb.pt")
    torch.save(c_model.state_dict(), dest + "/"+name+"_c.pt")
    torch.save(p_model.state_dict(), dest + "/"+name+"_p.pt")
    if best:
        print("!!!SAVING BEST!!!")
        torch.save(model.state_dict(), dest + "/BEST_bb.pt")
        torch.save(c_model.state_dict(), dest + "/BEST_c.pt")
        torch.save(p_model.state_dict(), dest + "/BEST_p.pt")


def toggle_downstream(c_model, p_model, switch=False):
    # switch - True if on, False if off
    if isinstance(c_model, torch.nn.DataParallel):
        c_model = c_model.module
        p_model = p_model.module
 
    for param in c_model.parameters():
        param.requires_grad_(switch)

    for param in p_model.parameters():
        param.requires_grad_(switch)

def toggle_backbone(backbone, switch=False):
    # switch - True if on, False if off
    if isinstance(backbone, torch.nn.DataParallel):
        backbone = backbone.module
 
    for param in backbone.parameters():
        param.requires_grad_(switch)
