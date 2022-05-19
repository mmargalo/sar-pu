import torch
from tqdm import tqdm
import numpy as np
from _utils import plot_grad_flow


def train_c_only(device, results_folder,
                backbone, c_model, p_model,
                train_loader, val_loader,
                criterion, optimizer, scheduler,
                train_writer, val_writer, 
                epochs=50):
    
    toggle(backbone, False)
    toggle(p_model, False)
    lowest_loss = 999.
        
    for epoch in range(epochs):
        print("C ONLY EPOCH: " + str(epoch))
        toggle(backbone, False)
        toggle(p_model, False)
        # train
        iter(epoch, device, results_folder, 
            backbone, c_model, p_model,
            train_loader, train_writer, criterion,
            optimizer, scheduler)
        ll = iter(epoch, device, results_folder, 
            backbone, c_model, p_model,
            val_loader, val_writer, criterion)

        if ll < lowest_loss:
            save(c_model, results_folder, best=True)
            print("BEST LL: " + str(ll))
            lowest_loss = ll

        save(c_model, results_folder)
  

def iter(epoch, device, results_folder,
        backbone, c_model, p_model,
        dataloader, writer, criterion, 
        optim=None, scheduler=None):

    is_train = False if optim is None else True

    backbone.train() if is_train else p_model.eval()
    c_model.train() if is_train else c_model.eval()
    p_model.train() if is_train else p_model.eval()

    loss_total = 0.
    per_class = np.array([0.,0.,0.])
    criterion = criterion()       
    
    for n_iter, (idx, x, labels) in enumerate(tqdm(dataloader)):
        x, labels = x.float().to(device), labels.float().to(device)
    
        with torch.set_grad_enabled(is_train):
            features = backbone(x).detach()
            exp_prior_y1 = c_model(features)
            exp_propensity = p_model(features).detach()

        if is_train: optim.zero_grad()
        loss = criterion(exp_prior_y1, exp_propensity, labels)
        loss, loss_total, per_class = update_loss(loss, loss_total, per_class)
        if is_train:
            loss.backward()
            plot_grad_flow(backbone, name="c_only", dest=results_folder)
            optim.step()

        # cycliclr
        if is_train:
            writer.add_scalar('lr_c', optim.param_groups[0]['lr'], epoch * len(dataloader) + n_iter)
            scheduler.step()

    writer.add_scalar('loss_conly', loss_total/len(dataloader), epoch)
    
    return loss_total/len(dataloader)



def toggle(model, switch=False):
    # switch - True if on, False if off
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
 
    for param in model.parameters():
        param.requires_grad_(switch)

def save(model, dest, name="checkpt", best=False):
    torch.save(model.state_dict(), dest + "/"+name+"_conly.pt")
    
    if best:
        print("!!!SAVING BEST!!!")
        torch.save(model.state_dict(), dest + "/BEST_conly.pt")
        
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