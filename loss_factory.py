import torch

class MultiMarginRank(torch.nn.Module):
    def __init__(self):
        super(MultiMarginRank, self).__init__()
        self.criterion = torch.nn.MarginRankingLoss(reduction='none')
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, pred, labels, weight=None):
        loss = torch.zeros_like(labels)
        for i in range(labels.shape[1]):
            s1 = labels[:,i]
            y1 = self.sigmoid(pred[:,i])
            for j in range(labels.shape[1]):
                s2 = labels[:,j]
                y2 = self.sigmoid(pred[:,j])
                loss[:, i] += self.criterion(y1, y2, s1-s2)
            loss[:,i] /= labels.shape[1]
            if weight is not None:
                loss[:,i] *= weight[:,i]
        return loss

class Loglikelihood(torch.nn.Module):
    def __init__(self):
        super(Loglikelihood, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, pred, prop, labels, weighted=False):        
        pred = torch.nan_to_num(self.sigmoid(pred))
        prop = torch.nan_to_num(self.sigmoid(prop))

        prob_labeled = pred*prop
        prob_unlabeled_pos = pred*(1-prop)
        prob_unlabeled_neg = 1-pred
        div = prob_unlabeled_pos+prob_unlabeled_neg
        div[div == 0] = 1e-4

        prob_pos_given_unl = prob_unlabeled_pos/div
        # prob_pos_given_unl = torch.nan_to_num(prob_unlabeled_pos/div)
        prob_neg_given_unl = 1-prob_pos_given_unl
        
        #prevent problems of taking log
        prob_labeled = prob_labeled + 1e-4
        prob_unlabeled_pos = prob_unlabeled_pos + 1e-4
        prob_unlabeled_neg = prob_unlabeled_neg + 1e-4

        if weighted:
            n_samples = labels.shape[0]
            n_classes = 2
            n_sample1 = torch.sum(labels, 0)
            n_samples = torch.add(torch.zeros_like(labels[0]), n_samples)
            n_sample0 = n_samples - torch.sum(labels, 0)
            pos = n_samples/(n_classes*n_sample1)
            neg = n_samples/(n_classes*n_sample0)
            
            pos = torch.nan_to_num(pos, nan=0.0, posinf=0, neginf=0)
            neg = torch.nan_to_num(neg, nan=0.0, posinf=0, neginf=0)

            weights = labels.clone().detach()
            weights2 = (weights.clone().detach()-1)*(-1)

            for n in range(3):
                weights[:,n] *= pos[n]
                weights2[:,n] *= neg[n]

            weights = weights + weights2

            ll = - weights *(
                labels*torch.log(prob_labeled)+
                (1-labels)*(
                    prob_pos_given_unl*torch.log(prob_unlabeled_pos)+
                    prob_neg_given_unl*torch.log(prob_unlabeled_neg))
            )

        else:
            ll = -(
                labels*torch.log(prob_labeled)+
                (1-labels)*(
                    prob_pos_given_unl*torch.log(prob_unlabeled_pos)+
                    prob_neg_given_unl*torch.log(prob_unlabeled_neg))
            )
            
        return ll

class BCELogits(torch.nn.Module):
    def __init__(self, weight=None, reduction="none"):
        super(BCELogits, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)
 
    def forward(self, pred, labels): 
        pred = torch.nan_to_num(pred)
        pred = pred + 1e-4      
        return self.criterion(pred, labels)

class COnly(torch.nn.Module):
    def __init__(self, reduction="none"):
        super(COnly, self).__init__()
        self.reduction = reduction
         
    def forward(self, x, e, s):
        weights_pos = s/e
        weights_neg = (1-s) + s*(1-1/e)

        Xp = torch.cat((x, x), 0)
        Yp = torch.cat((torch.ones_like(s), torch.zeros_like(s)), 0)
        Wp = torch.cat((weights_pos, weights_neg), 0)
        
        criterion = torch.nn.BCEWithLogitsLoss(weight=Wp, reduction=self.reduction)
        return criterion(Xp, Yp)


loss_dict = {
    'bcelogits': BCELogits,
    'loglikelihood': Loglikelihood,
    'multimargin': MultiMarginRank,
    'conly': COnly
}


def get_loss(loss_name):
    assert loss_name in loss_dict.keys()
    return loss_dict[loss_name]
