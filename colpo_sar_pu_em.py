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
    for epoch in range(outer_epochs):
        print("EPOCH: " + str(epoch))
        backbone_loop(epoch, train_loader, train_writer, results_folder,
                    backbone, c_model, p_model,
                    criterion, c_criterion, p_criterion,
                    optim, c_optim, p_optim, 
                    inner_epochs=inner_epochs)
        ll = backbone_loop(epoch, val_loader, val_writer, results_folder,
                    backbone, c_model, p_model, 
                    criterion, c_criterion, p_criterion,
                    scheduler=scheduler, c_scheduler=c_scheduler, p_scheduler=p_scheduler)

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
                inner_epochs=1, device='cuda'):

    is_train = False if optim is None else True

    toggle_downstream(c_model, p_model, True)
    downstream_loop(outer_epoch, inner_epochs, writer, 
                    results_folder, dataloader,
                    backbone, c_model, p_model, 
                    c_criterion, p_criterion, 
                    c_optim, p_optim,
                    c_scheduler, p_scheduler)
    toggle_downstream(c_model, p_model, False)
    
    backbone.train() if is_train else backbone.eval()

    loss_total = 0.
    per_class = np.array([0.,0.,0.])
    criterion = criterion()
    for n_iter, (x, labels) in enumerate(tqdm(dataloader)):
        x, labels = x.float().to(device), labels.float().to(device)
        
        with torch.set_grad_enabled(is_train):
            features = backbone(x)

            exp_prior_y1 = c_model(features)
            exp_propensity = p_model(features)

            if is_train: optim.zero_grad()
            loss = criterion(exp_prior_y1, exp_propensity, labels)
            loss, loss_total, per_class = update_loss(loss, loss_total, per_class)
            if is_train:
                loss.backward()
                plot_grad_flow(backbone, name="backbone", dest=results_folder)
                optim.step()

    ll = loss_total/len(dataloader)

    if is_train: 
        writer.add_scalar('lr', optim.param_groups[0]['lr'], outer_epoch)
    else:
        scheduler.step(ll)
    
    writer.add_scalar('log_likelihood', ll, outer_epoch)
    for l_i, l in enumerate(per_class):
        writer.add_scalar('log_likelihood_'+str(l_i), l/len(dataloader), outer_epoch)

    return ll


def downstream_loop(outer_epoch, epochs, writer, 
                    results_folder, dataloader, 
                    backbone, c_model, p_model, 
                    c_criterion, p_criterion, 
                    c_optim, p_optim,
                    c_scheduler, p_scheduler, device='cuda'):
                    
    is_train = False if c_optim is None and p_optim is None else True

    c_model.train() if is_train else c_model.eval()
    p_model.train() if is_train else p_model.eval()

    c_criterion = c_criterion(reduction='none')

    for epoch in range(epochs):
        c_loss_total = 0.
        p_loss_total = 0.
        per_c_loss = np.array([0.,0.,0.])
        per_p_loss = np.array([0.,0.,0.])

        for n_iter, (x, s) in enumerate(tqdm(dataloader)):
            x, s = x.float().to(device), s.float().to(device)
            
            with torch.set_grad_enabled(is_train):
                features = backbone(x).detach()

                exp_prior_y1 = c_model(features)
                exp_propensity = p_model(features)
                exp_post_y1 = expectation_y(exp_prior_y1, exp_propensity, s) 

                if is_train:
                    c_optim.zero_grad()
                    p_optim.zero_grad()

                # always need to update weights
                p_criterion2 = p_criterion(weight=exp_post_y1.detach(), reduction='none')
                p_loss = p_criterion2(exp_propensity, s)
                p_loss, p_loss_total, per_p_loss = update_loss(p_loss, p_loss_total, per_p_loss)

                c_loss = c_criterion(exp_prior_y1, exp_post_y1)
                c_loss, c_loss_total, per_c_loss = update_loss(c_loss, c_loss_total, per_c_loss)

                if is_train:
                    p_loss.backward(retain_graph=True)
                    plot_grad_flow(p_model, name="propensity", dest=results_folder)
                    c_loss.backward()
                    plot_grad_flow(c_model, name="classifier", dest=results_folder)
                    p_optim.step()
                    c_optim.step()
                
        if is_train:
            writer.add_scalar('lr_classifier', c_optim.param_groups[0]['lr'], outer_epoch*(epoch+1))
            writer.add_scalar('lr_propensity', p_optim.param_groups[0]['lr'], outer_epoch*(epoch+1))
        else:
            c_scheduler.step(c_loss_total/len(dataloader))
            p_scheduler.step(p_loss_total/len(dataloader))

        writer.add_scalar('loss_classifier', c_loss_total/len(dataloader), outer_epoch*(epoch+1))
        writer.add_scalar('loss_propensity', p_loss_total/len(dataloader), outer_epoch*(epoch+1))
        for c_i, c in enumerate(per_c_loss):
            writer.add_scalar('loss_classifier_'+str(c_i), c/len(dataloader), outer_epoch*(epoch+1))
        for p_i, p in enumerate(per_p_loss):
            writer.add_scalar('loss_propensity_'+str(p_i), p/len(dataloader), outer_epoch*(epoch+1))


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