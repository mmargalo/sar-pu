import torch

class Loglikelihood(torch.nn.Module):
    def __init__(self):
        super(Loglikelihood, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, pred, prop, labels):        
        pred = torch.nan_to_num(self.sigmoid(pred))
        prop = torch.nan_to_num(self.sigmoid(prop))

        prob_labeled = pred*prop
        prob_unlabeled_pos = pred*(1-prop)
        prob_unlabeled_neg = 1-pred
        div = prob_unlabeled_pos+prob_unlabeled_neg
        div[div == 0] = 0.00000001

        prob_pos_given_unl = prob_unlabeled_pos/div
        # prob_pos_given_unl = torch.nan_to_num(prob_unlabeled_pos/div)
        prob_neg_given_unl = 1-prob_pos_given_unl
        
        #prevent problems of taking log
        prob_labeled = prob_labeled + 0.00000001
        prob_unlabeled_pos = prob_unlabeled_pos + 0.00000001
        prob_unlabeled_neg = prob_unlabeled_neg + 0.00000001

        ll = -(
            labels*torch.log(prob_labeled)+
            (1-labels)*(
                prob_pos_given_unl*torch.log(prob_unlabeled_pos)+
                prob_neg_given_unl*torch.log(prob_unlabeled_neg))
        )
        return ll


loss_dict = {
    'bcelogits': torch.nn.BCEWithLogitsLoss,
    'loglikelihood': Loglikelihood
}


def get_loss(loss_name):
    assert loss_name in loss_dict.keys()
    return loss_dict[loss_name]
