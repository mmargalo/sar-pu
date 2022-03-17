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

# for debugging
torch.autograd.set_detect_anomaly(True)

_FRZ_C = False
_FRZ_P = False

_DEVICE = None

def train_sar_em(model,
        device,
        train_dataloader,
        val_dataloader,
        max_its=100,
        slope_eps=0.0001,
        ll_eps=0.0001,
        convergence_window=10,
        refit_classifier=True,
        lr=0.01,
        momentum=0.9,
        train_writer=None, 
        val_writer=None,
        results_folder=None,
        prop_weight=1.
        ):

    """
        SAR-PU Expectation Maximization using NN
        :param model: NN model
        :param device: cuda or cpu
        :param train_dataloader: training set dataloader 
        :param val_dataloader: validation set dataloader
        :param max_its: maximum training iterations
        :param slope_eps: degree of change to consider, log likelihood (converge)
        :param ll_eps: log likelihood to consider (converge)
        :param convergence_window: check if converged after N epochs
        :param refit_classifier: refit classifier on best weights
        :param lr: learning rate
        :param momentum: optimizer momentum
        :param train_writer: Tensorboard training writer
        :param val_writer: Tensorboard validation writer
        :param results_folder: folder to store results
        :return: best model weights
    """

    global _DEVICE
    global _FRZ_C
    global _FRZ_P

    _DEVICE = device

    optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=momentum)
    #optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    
    train_s = np.array(train_dataloader.dataset.get_labels(), dtype='float64')
    val_s = np.array(val_dataloader.dataset.get_labels(), dtype='float64')

    # init
    epochs_without_imp_c = 0 
    epochs_without_imp_p = 0 
    epochs_without_imp = 0 
    lowest_c, lowest_p = 999, 999
    highest_vll = -999
    best_exp_prop = None

    # store scores
    _FRZ_P = False
    _FRZ_C = False
    train_lists = None
    val_lists = None
    val_ll = None

    for i in range(0, max_its):
        #pos_weight = train_dataloader.dataset.get_pos_weight()
        pos_weight = None
        c_loss, p_loss, train_lists = train(model, device=device, train_loader=train_dataloader, optimizer=optimizer, epoch=i, lists=train_lists, writer=train_writer, pos_weight=pos_weight, prop_weight=prop_weight)
        exp_prior_y1, _, exp_prop = train_lists
        exp_prior_y1 = np.array(exp_prior_y1, dtype='float64')
        exp_prop = np.array(exp_prop, dtype='float64')

        val_c_loss, val_p_loss, val_lists = val(model, device=device, val_loader=val_dataloader, epoch=i, lists=val_lists, writer=val_writer, pos_weight=pos_weight, prop_weight=prop_weight)
        val_exp_prior_y1, _, val_exp_prop = val_lists
        val_exp_prior_y1 = np.array(val_exp_prior_y1, dtype='float64')
        val_exp_prop = np.array(val_exp_prop, dtype='float64')

        ll = loglikelihood_probs(exp_prior_y1, exp_prop, train_s)
        #train_loglikelihoods.append(ll)
        for ll_i, ll_res in enumerate(ll):
            train_writer.add_scalar('log_likelihood_' +  str(ll_i), ll_res, i)
        ll = np.mean(ll)
        train_writer.add_scalar('log_likelihood', ll, i)
        print("train-ll",ll)

        val_ll = loglikelihood_probs(val_exp_prior_y1, val_exp_prop, val_s)
        #val_loglikelihoods.append(val_ll)
        for ll_i, ll_res in enumerate(val_ll):
            val_writer.add_scalar('log_likelihood_' +  str(ll_i), ll_res, i)
        val_ll = np.mean(val_ll)
        print("val-ll",val_ll)
        
        val_writer.add_scalar('log_likelihood', val_ll, i)
        train_writer.add_scalar('combined_loss', c_loss+p_loss, i)
        val_writer.add_scalar('combined_loss', val_c_loss+val_p_loss, i)

        val_writer.add_scalar('epochs_without_imp', epochs_without_imp, i)
        val_writer.add_scalar('epochs_without_imp_c', epochs_without_imp_c, i)
        val_writer.add_scalar('epochs_without_imp_p', epochs_without_imp_p, i)

        train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], i)

        if highest_vll < val_ll:
            highest_vll = val_ll
            best_exp_prop = deepcopy(exp_prop)
            save(model, results_folder, best=True)
       
        print("Current LR : " + str(optimizer.param_groups[0]['lr']))   
        scheduler.step(val_ll)

        if optimizer.param_groups[0]['lr'] < 0.001:
            if lowest_c > val_c_loss:
                lowest_c = val_c_loss
                epochs_without_imp_c = 0
            else:
                epochs_without_imp_c += 1
                if epochs_without_imp_c == 5 and not _FRZ_C:
                    epochs_without_imp_c = 0
                    print("--Freezing Classifier--")
                    toggle_fc(model, off='classifier')
                    _FRZ_C = True
                    train_writer.add_scalar('froze_epoch_c', i, 0)
            
            if lowest_p > val_p_loss:
                lowest_p = val_p_loss
                epochs_without_imp_p = 0
            else:
                epochs_without_imp_p += 1
                if epochs_without_imp_p == 5 and not _FRZ_P:
                    epochs_without_imp_p = 0
                    print("--Freezing Propensity--")
                    toggle_fc(model, off='propensity')
                    _FRZ_P = True
                    val_writer.add_scalar('froze_epoch_p', i, 0)
    
            if _FRZ_C and _FRZ_P:
                save(model, results_folder)
                print("FROZEN")
                break #stopped learning


        save(model, results_folder)
    save_yaml(results_folder, "best_exp_prop", best_exp_prop)
    i = 0
    pos_weight = None
    model.load_state_dict(torch.load(results_folder + "/BEST_checkpoint.pt"))

    if refit_classifier:
        print("Refitting classifier...")
        for g in optimizer.param_groups:
            g['lr'] = 0.001
        exp_prop = load_yaml(results_folder, "best_exp_prop")
        train_c_only(model, exp_prop, device, train_dataloader, optimizer, i, writer=train_writer, pos_weight=None)
        save(model, results_folder, name="refit")

    return model


def save(model, dest, name="checkpoint", best=False):
    torch.save(model.state_dict(), dest + "/"+name+".pt")
    if best:
        print("!!!SAVING BEST!!!")
        torch.save(model.state_dict(), dest + "/BEST_checkpoint.pt")


def push(array_queue, new_array):
    array_queue[:,:-1] = array_queue[:,1:]
    array_queue[:,-1] = new_array


def adjust_learning_rate(optimizer, shrink_factor=0.1, manual=None):
    print("DECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor if manual is None else manual
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def get_list(idx, list):
    out = []
    for i in idx:
        i = i.item()
        out.append(list[i])
    return torch.FloatTensor(out)


def train(model, device, train_loader, optimizer, epoch, lists=None, writer=None, pos_weight=None, prop_weight=1.):
   
    model.train()
    sigmoid = torch.nn.Sigmoid()

    c_loss_total = 0
    p_loss_total = 0

    per_class_loss = np.array([0.,0.,0.])
    per_prop_loss = np.array([0.,0.,0.])

    print("TRAIN-EPOCH: " + str(epoch))

    if lists is None:
        # for epoch 0
        size = len(train_loader.dataset)
        y1_prior_list = [None] * size
        y1_post_list = [None] * size
        prop_list = [None] * size
    else:
        y1_prior_list, y1_post_list, prop_list = lists

    for batch_idx, (idx, x, s, _) in enumerate(tqdm(train_loader)):
        param_c, param_p = {}, {}
        x, s = x.float().to(device), s.float().to(device)
        #pos_weight = (len(s)-torch.sum(s))/torch.sum(s)
        if epoch == 0: 
            # INITIALIZE –do an entire pass on the whole training dataset
            proportion_labeled = torch.sum(s, 0)/len(s)
            # invert proportions as weights for labeled and unlabeled
            c_weights = s * (1 - proportion_labeled) + (1 - s) * proportion_labeled
            
            c_out, p_out = fit(model, x, s)
            c_exp = sigmoid(c_out).view_as(s).detach()
            param_c['pred'] = c_out
            param_c['target'] = s
            param_c['weights'] = c_weights
            param_p['pred'] = p_out
            param_p['target'] = s
            param_p['weights'] = s + (1 - s) * c_exp

            c_loss, c_per, p_loss, p_per = get_loss(model, param_c, param_p, optimizer=optimizer, pos_weight=pos_weight, prop_weight=prop_weight)

        else:
            # retrieve based on idx -new set of indices per iter
            y1 = get_list(idx, y1_post_list).to(device)
            
            c_out, p_out = fit(model, x, s)
            param_c['pred'] = c_out
            param_c['target'] = y1
            param_c['weights'] = None
            param_p['pred'] = p_out
            param_p['target'] = s
            param_p['weights'] = y1
            
            c_loss, c_per, p_loss, p_per = get_loss(model, param_c, param_p, optimizer=optimizer, pos_weight=pos_weight, prop_weight=prop_weight)

        c_loss_total += c_loss
        p_loss_total += p_loss
        per_class_loss = per_class_loss + np.array(c_per) 
        per_prop_loss = per_prop_loss + np.array(p_per) 

        exp_prior_y1, exp_prop = fit(model, x, s, train=False)
        exp_post_y1 = expectation_y(sigmoid(exp_prior_y1), sigmoid(exp_prop), s)
        exp_post_y1 = torch.nan_to_num(exp_post_y1, posinf=1.0)
        # save all classes
        y1_prior_list = set_list(idx, sigmoid(exp_prior_y1), y1_prior_list)
        prop_list = set_list(idx, sigmoid(exp_prop), prop_list)
        y1_post_list = set_list(idx, exp_post_y1, y1_post_list)
    
    batch_count = len(train_loader)
    writer.add_scalar('loss_classifier', c_loss_total/(batch_count), epoch)
    writer.add_scalar('loss_propensity', p_loss_total/(batch_count), epoch)

    for c_i, c in enumerate(per_class_loss):
        writer.add_scalar('loss_classifier_'+str(c_i), c/(batch_count), epoch)
    for p_i, p in enumerate(per_prop_loss):
        writer.add_scalar('loss_propensity_'+str(p_i), p/(batch_count), epoch)

    assert None not in y1_prior_list
    assert None not in y1_post_list
    assert None not in prop_list

    assert math.inf not in y1_prior_list
    assert math.nan not in y1_prior_list
    assert math.inf not in prop_list
    assert math.nan not in prop_list

    return c_loss_total/batch_count, p_loss_total/batch_count, (y1_prior_list, y1_post_list, prop_list)


def train_c_only(model, exp_prop, device, train_loader, optimizer, epoch, writer=None, pos_weight=None):
    global _FRZ_C
    #adjust_learning_rate(optimizer, manual=0.0001)
    _FRZ_C = False
    toggle_fc(model, off='propensity')
    
    c_loss_total = 0
    
    print("C ONLY EPOCH: " + str(epoch))
    param_c = {}
        
    for batch_idx, (idx, x, s, _) in enumerate(tqdm(train_loader)):
        x, s = x.to(device), s.to(device)

        c_out, p_out = fit(model, x, s)
        param_c['pred'] = c_out
        param_c['target'] = s
        param_c['weights'] = None
        c_loss, _ = get_loss(model, param_c, None, optimizer=optimizer, pos_weight=pos_weight, e=get_list(idx, exp_prop).to(device))
        c_loss_total += c_loss
    
    writer.add_scalar('loss_classifier_only', c_loss_total/len(train_loader), epoch)

    return c_loss_total/len(train_loader)

def val(model, device, val_loader, epoch, lists=None, writer=None, pos_weight=None, prop_weight=1.):
    
    model.eval()
    sigmoid = torch.nn.Sigmoid()

    c_loss_total = 0
    p_loss_total = 0

    per_class_loss = np.array([0.,0.,0.])
    per_prop_loss = np.array([0.,0.,0.])

    print("VAL-EPOCH: " + str(epoch))

    if lists is None:
        size = len(val_loader.dataset)
        y1_prior_list = [None] * size
        y1_post_list = [None] * size
        prop_list = [None] * size
    else:
        y1_prior_list, y1_post_list, prop_list = lists
    
    with torch.no_grad():
        for batch_idx, (idx, x, s, _) in enumerate(tqdm(val_loader)):
            param_c, param_p = {}, {}
            x, s = x.float().to(device), s.float().to(device)
            #pos_weight = (len(s)-torch.sum(s))/torch.sum(s)
            if epoch == 0: 
                # INITIALIZE –do an entire pass on the whole training dataset
                proportion_labeled = torch.sum(s, 0)/len(s)
                # invert proportions as weights for labeled and unlabeled
                c_weights = s * (1 - proportion_labeled) + (1 - s) * proportion_labeled
                
                c_out, p_out = fit(model, x, s, train=False)
                c_exp = sigmoid(c_out).view_as(s).detach()
                param_c['pred'] = c_out
                param_c['target'] = s
                param_c['weights'] = c_weights
                param_p['pred'] = p_out
                param_p['target'] = s
                param_p['weights'] = s + (1 - s) * c_exp

                c_loss, c_per, p_loss, p_per = get_loss(model, param_c, param_p, pos_weight=pos_weight, prop_weight=prop_weight)

            else:
                # retrieve based in idx -new set of indices per iter
                y1 = get_list(idx, y1_post_list).to(device)

                c_out, p_out = fit(model, x, s, train=False)
                param_c['pred'] = c_out
                param_c['target'] = y1
                param_c['weights'] = None
                param_p['pred'] = p_out
                param_p['target'] = s
                param_p['weights'] = y1

                c_loss, c_per, p_loss, p_per = get_loss(model, param_c, param_p, pos_weight=pos_weight, prop_weight=prop_weight)

            c_loss_total += c_loss
            p_loss_total += p_loss
            per_class_loss = per_class_loss + np.array(c_per) 
            per_prop_loss = per_prop_loss + np.array(p_per) 

            exp_prior_y1, exp_prop = fit(model, x, s, train=False)
            exp_post_y1 = expectation_y(sigmoid(exp_prior_y1), sigmoid(exp_prop), s)
            exp_post_y1 = torch.nan_to_num(exp_post_y1, posinf=1.0)
            # save all classes
            y1_prior_list = set_list(idx, sigmoid(exp_prior_y1), y1_prior_list)
            prop_list = set_list(idx, sigmoid(exp_prop), prop_list)
            y1_post_list = set_list(idx, exp_post_y1, y1_post_list)

    batch_count = len(val_loader)

    writer.add_scalar('loss_classifier', c_loss_total/batch_count, epoch)
    writer.add_scalar('loss_propensity', p_loss_total/batch_count, epoch)

    for c_i, c in enumerate(per_class_loss):
        writer.add_scalar('loss_classifier_'+str(c_i), c/(batch_count), epoch)

    for p_i, p in enumerate(per_prop_loss):
        writer.add_scalar('loss_propensity_'+str(p_i), p/(batch_count), epoch)


    assert None not in y1_prior_list
    assert None not in y1_post_list
    assert None not in prop_list

    assert math.inf not in y1_prior_list
    assert math.nan not in y1_prior_list
    assert math.inf not in y1_post_list
    assert math.nan not in y1_post_list
    assert math.inf not in prop_list
    assert math.nan not in prop_list

    return c_loss_total/batch_count, p_loss_total/batch_count, (y1_prior_list, y1_post_list, prop_list)


def fit(model, input, label, train=True):

    model.train()
    if not train: model.eval()
    return model(input, label)


# lol, get it?
def get_loss(model, param_c, param_p, optimizer=None, pos_weight=None, e=None, prop_weight=1.): 

    if e is not None:
        x = param_c['pred']
        s = param_c['target']
        weights_pos = s/e
        weights_neg = (1-s) + s*(1-1/e)
    
        Xp = torch.cat((x, x), 0)
        Yp = torch.cat((torch.ones_like(s), torch.zeros_like(s)), 0)
        Wp = torch.cat((weights_pos, weights_neg), 0)
        loss_c, per_c = thru_loss(Xp, Yp, weights=Wp, pos_weight=pos_weight)

        loss_c.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss_c.item(), per_c

    else:
        if param_c['weights'] is None:
            c_pred = torch.cat((param_c['pred'], param_c['pred']), 0)
            c_target = torch.cat((torch.ones_like(param_c['target']), torch.zeros_like(param_c['target'])), 0)
            c_weight = torch.cat((param_c['target'], 1-param_c['target']), 0)
            loss_c, per_c = thru_loss(c_pred, c_target, weights=c_weight, pos_weight=pos_weight)
        else:
            loss_c, per_c = thru_loss(param_c['pred'], param_c['target'], weights=param_c['weights'], pos_weight=pos_weight)
        loss_p, per_p = thru_loss(param_p['pred'], param_p['target'], weights=param_p['weights'], pos_weight=pos_weight)


        #loss_p, per_p = thru_loss(param_p['pred'], param_p['target'], weights=param_p['weights'], pos_weight=pos_weight)
        loss = loss_c + (prop_weight * loss_p)

        if optimizer:
            #try:
            loss.backward()
            plot_grad_flow(model)
            optimizer.step()
            optimizer.zero_grad()
            #except:
            #    print("===Backward Error===")
            #    optimizer.zero_grad()
            #    return None

        return loss_c.item(), per_c, loss_p.item(), per_p


def thru_loss(pred, target, weights=None, pos_weight=None, reduction='none'):
    
    criterion = torch.nn.BCEWithLogitsLoss(weight=weights, pos_weight=pos_weight, reduction=reduction)
    loss_tensor = criterion(pred.view_as(target).clamp(min=1e-4), target.float())
    # per class loss
    loss_per = [l.item() for l in torch.mean(loss_tensor, 0)]

    return loss_tensor.mean(), loss_per


def set_list(idx, items, list):
    
    for i, item in zip(idx, items.detach().clone()):
        detached = [x.item() for x in item]
        list[i.item()] = detached
    return list


def test_sar_em(model, device, test_loader):

    c_outputs = None
    p_outputs = None
    sigmoid = torch.nn.Sigmoid()
    
    print("---TEST---")

    with torch.no_grad():
        for batch_idx, (_, x, s, _) in enumerate(tqdm(test_loader)):
            x = x.to(device)
            c_output, p_output = fit(model, x, train=False)
            c_output = sigmoid(c_output)
            p_output = sigmoid(p_output)
            #y1 = expectation_y(c_output, p_output, target)

            c_output_arr = c_output.data.cpu().numpy().astype("float64")
            p_output_arr = p_output.data.cpu().numpy().astype("float64")
            if c_outputs is None:
                c_outputs = c_output_arr
            else:
                c_outputs = np.concatenate((c_outputs, c_output_arr), axis=0)

            if p_outputs is None:
                p_outputs = p_output_arr
            else:
                p_outputs = np.concatenate((p_outputs, p_output_arr), axis=0)

    return c_outputs, p_outputs


def expectation_y(expectation_f,expectation_e, s):
    #s = s.unsqueeze(1)
    result= s + (1-s) * (expectation_f*(1-expectation_e))/(1-expectation_f*expectation_e)
    return result

#  Expected loglikelihood of the model probababilities
def loglikelihood_probs(class_probabilities, propensity_scores, labels):

    class_probabilities = np.nan_to_num(class_probabilities)
    propensity_scores = np.nan_to_num(propensity_scores)

    prob_labeled = class_probabilities*propensity_scores
    prob_unlabeled_pos = class_probabilities*(1-propensity_scores)
    prob_unlabeled_neg = 1-class_probabilities
    div = prob_unlabeled_pos+prob_unlabeled_neg
    div[div == 0] = 0.00000001
    prob_pos_given_unl = np.nan_to_num(prob_unlabeled_pos/div)
    prob_neg_given_unl = 1-prob_pos_given_unl

    #prevent problems of taking log
    prob_labeled[prob_labeled == 0] = 0.00000001
    prob_unlabeled_pos[prob_unlabeled_pos == 0] = 0.00000001
    prob_unlabeled_neg[prob_unlabeled_neg == 0] = 0.00000001

    assert np.nan not in prob_labeled
    assert np.nan not in prob_unlabeled_pos
    assert np.nan not in prob_unlabeled_neg
    
    ll = (
        labels*np.log(prob_labeled)+
        (1-labels)*(
            prob_pos_given_unl*np.log(prob_unlabeled_pos)+
            prob_neg_given_unl*np.log(prob_unlabeled_neg))
    )
    #ll_cancer = labels[:,2]*np.log(class_probabilities[:,2])+(1-labels[:,2])*np.log(1-class_probabilities[:,2])
    
    ll = np.mean(ll, axis=0)
    #ll_cancer = np.mean(ll_cancer)

    #return np.append(ll, ll_cancer)
    return ll

def slope(array, axis=0):
    """Calculate the slope of the values in ar over dimension "axis". The values are assumed to be equidistant."""
    if axis==1:
        array = array.transpose()

    n = array.shape[0]
    norm_x = np.asarray(range(n))-(n-1)/2
    auto_cor_x = np.square(norm_x).mean(0)
    avg_y = array.mean(axis=0)
    norm_y = array - avg_y
    cov_x_y = np.matmul(norm_y.transpose(),norm_x)/n
    result = cov_x_y/auto_cor_x
    if axis==1:
        result = result.transpose()
    return result


def save_yaml(results_folder, name, data):

    name = name + '.yaml'
    with open(join(results_folder, name), 'w') as f:
        yaml.dump(data.tolist(), f)
        print("Successfully saved " + name)

def load_yaml(results_folder, name):

    name = name + '.yaml'
    with open(join(results_folder, name)) as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def toggle_fc(model, off):

    mod = model.module if _DEVICE == 'cuda' and torch.cuda.device_count() > 1 else model
    layers = [1,4,6]

    try:
        if off == 'classifier':
            for layer in layers:
                mod.fc_classifier[layer].bias.requires_grad = False
                mod.fc_classifier[layer].weight.requires_grad = False

        elif off == 'propensity':
            for layer in layers:
                mod.fc_propensity[layer].bias.requires_grad = False
                mod.fc_propensity[layer].weight.requires_grad = False
    except:

        if off == 'classifier':
            mod.fc_classifier.bias.requires_grad = False
            mod.fc_classifier.weight.requires_grad = False
            
        elif off == 'propensity':
            mod.fc_propensity.bias.requires_grad = False
            mod.fc_propensity.weight.requires_grad = False