import argparse
from dataset.dataset_pu import PUMultiLabel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import logistic
from model2 import convert_weights
from tqdm import tqdm
import os
import sys
import datetime
from os.path import join
from score import score_threshold, score_fuzzy
from setup.setup_multirun import nondeterministic
from model.encoder.model_factory import get_model
from setup.setup_logs import setup_logs
from setup.setup_loaders import setup_loaders
from utils import adjust_learning_rate
from loss.pu_rank_loss import PURankLoss
import math

def expectation_y(expectation_f, expectation_e, s):
   #print("EXPECTATION F")
   #print(expectation_f)
   #print("EXPECTATION E")
   #print(expectation_e)
   #print("LABELS")
   #print(s)
    result= s + (1-s) * (expectation_f*(1-expectation_e))/(1-expectation_f*expectation_e)
    return result

def train(args, c_model, p_model, device, train_loader, c_optim, p_optim, epoch, writer):
    c_model.train()
    p_model.train()
    outputs = None
    labels = None
    c_loss_total = 0
    p_loss_total = 0
    c_criterion = nn.BCELoss()
    sigmoid = nn.Sigmoid()
        
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        c_optim.zero_grad()
        p_optim.zero_grad()
        c_output = sigmoid(c_model(data))
        p_output = sigmoid(p_model(data))
        y1 = expectation_y(c_output, p_output, target)

        p_criterion = nn.BCELoss(weight=y1.detach())

        with torch.no_grad():
            output_arr = c_output.data.cpu().numpy()
            #output_arr = logistic.cdf(output_arr)
            if labels is None:
                labels = target.data.cpu().numpy()
                outputs = output_arr
            else:
                labels = np.concatenate((labels, target.data.cpu().numpy()), axis=0)
                outputs = np.concatenate((outputs, output_arr), axis=0)
        
        c_loss = c_criterion(c_output, y1.detach())
        p_loss = p_criterion(p_output, target)

        c_loss_total += c_loss.item()
        p_loss_total += p_loss.item()
        c_loss.backward()
        p_loss.backward()
        c_optim.step()
        p_optim.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tC_Loss: {:.6f}\tP_Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), c_loss.item(), p_loss.item()))
            #writer.add_scalar('loss_mini', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('lr_classifier', c_optim.param_groups[0]['lr'],
                              epoch * len(train_loader) + batch_idx)
            writer.add_scalar('lr_propensity', p_optim.param_groups[0]['lr'],
                              epoch * len(train_loader) + batch_idx)

    score_threshold(epoch, outputs, labels, writer, args.threshold)
    score_fuzzy(epoch, outputs, labels, writer)
    writer.add_scalar('loss_classifier', c_loss_total/batch_idx, epoch)
    writer.add_scalar('loss_propensity', p_loss_total/batch_idx, epoch)

def test(args, model, device, test_loader, epoch=None, writer=None):
    outputs = None
    labels = None
    with torch.no_grad():
        for data, target in tqdm(test_loader, total=len(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_arr = output.data.cpu().numpy()
            output_arr = logistic.cdf(output_arr)

            if labels is None:
                labels = target.data.cpu().numpy()
                outputs = output_arr
            else:
                labels = np.concatenate((labels, target.data.cpu().numpy()),
                                        axis=0)
                outputs = np.concatenate((outputs, output_arr),
                                         axis=0)

    f1 = score_threshold(epoch, outputs, labels, writer, args.threshold)
    precision = score_fuzzy(epoch, outputs, labels, writer)

    return precision

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-class_count', type=int, default=1, metavar='N',
                        help='input number of classes')
    parser.add_argument('-batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('-epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('-momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('-log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-threshold', type=float, default=0.5,
                        help='threshold for the evaluation (default: 0.5)')
    parser.add_argument('-image_path', help='path for the image folder')
    parser.add_argument('-dataset_path', help='path for text file with paths and labels')
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-snapshot', default=None)
    parser.add_argument('-resume', type=int, default=None)
    parser.add_argument('-test_model', action='store_true')
    parser.add_argument('-save_path')
    parser.add_argument('-model', help='<base model name>-<type>')
    args = parser.parse_args()

    # call to execute uniform runs
    nondeterministic()
    device = "cpu"
    
    if not args.test_model:
        train_writer, val_writer = setup_logs(args, overwrite=True)
        train_loader, test_loader = setup_loaders(args, PU=True)
    else:
        test_loader = setup_loaders(args)
 
    model = args.model.split('-')
    classifier_model = get_model(model[0], type=model[1], class_count=args.class_count, mode='basic').to(device)
    propensity_model = get_model(model[0], type=model[1], class_count=args.class_count, mode='basic').to(device)
    
    #if torch.cuda.device_count() > 1:
    #    classifier_model = nn.DataParallel(classifier_model)
    #    propensity_model = nn.DataParallel(propensity_model)
    
    #if args.snapshot:
    #    if isinstance(model, nn.DataParallel):
    #        model.load_state_dict(torch.load(args.snapshot))
    #    else:
    #        model.load_state_dict(convert_weights(torch.load(args.snapshot)))
    #    if not args.test_model:
    #        assert args.resume is not None
    #        resume = args.resume
    #        print("Resuming at", resume)
    #    else:
    #        resume = 0
    #else:
    resume = 1
    highest_prec = 0
    epochs_without_imp = 0

    if not args.test_model:
        c_optim = optim.SGD(classifier_model.parameters(), lr=args.lr,
                              momentum=args.momentum)
        p_optim = optim.SGD(propensity_model.parameters(), lr=args.lr,
                              momentum=args.momentum)
        # base_optim = optim.Adam(model.parameters(), lr=args.lr)
        #optimizer = base_optim
    for epoch in range(resume, resume+args.epochs + 1):
        if not args.test_model:
            train(args, classifier_model, propensity_model, device, train_loader, c_optim, p_optim, epoch, train_writer)
            prec = test(args, classifier_model, device, test_loader, epoch, val_writer)
            torch.save(classifier_model.state_dict(), args.save_path + "/checkpoint_c.pt")
            torch.save(propensity_model.state_dict(), args.save_path + "/checkpoint_p.pt")
            if prec > highest_prec:
                torch.save(classifier_model.state_dict(), args.save_path + "/BEST_checkpoint_c.pt")
                torch.save(propensity_model.state_dict(), args.save_path + "/BEST_checkpoint_p.pt")
                print("Now the highest precision is %.2f%%, it was %.2f%%" % (
                    100*prec, 100*highest_prec))
                highest_prec = prec
            else:
                epochs_without_imp += 1
                print("Highest precision is still %.2f%%, epochs without imp. %d" % (
                    100 * highest_prec, epochs_without_imp))
                if epochs_without_imp == 3:
                    adjust_learning_rate(c_optim, 0.1)
                    adjust_learning_rate(p_optim, 0.1)
                    epochs_without_imp = 0
            #for param_group in c_optim.param_groups:
            #    param_group['lr'] = args.lr/math.sqrt(epoch)
            #for param_group in p_optim.param_groups:
            #    param_group['lr'] = args.lr/math.sqrt(epoch)
        else:
            prec = test(args, classifier_model, device, test_loader)
            break

if __name__ == '__main__':
    main()
