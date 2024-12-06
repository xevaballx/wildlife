

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers, compute each alpha
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return: per_cls_weights: 
    The idea is to discount the contribution of classes with many samples 
    (i.e., frequent classes) using the parameter 𝛽.
    This introduces a smoothing effect that reduces the influence of very large n.
    Common choices for β are 0.99, 0.999, or 0.9999. With β close to 1, rare 
    classes are given more importance
    """
    # effective number (1 - beta^n) / (1 - beta)
    effective_num = (1.0 - np.power(beta, cls_num_list))/(1.0 - beta)
    alpha = 1.0 / effective_num

     # normalize the weights so they sum to the number of classes
    per_cls_weights = alpha / np.sum(alpha) * len(cls_num_list)


    return torch.FloatTensor(per_cls_weights)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0, device='cpu'):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma

        if weight == "None": weight = None
        self.weight = weight # output from reweight
        self.device=device
        

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions (logits)
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        if self.weight is not None:
            # print("in first self weight", self.weight)
            self.weight = self.weight.to(input.device)

        # compute softmax probabilities
        probs = F.softmax(input, dim=1) # (N, num_classes)

        # probs of the target class
        # target = target.long() 
        # target_one_hot = F.one_hot(target, num_classes=probs.size(1)).float()

        # print("probs shape:", probs.shape)  # (N, C)
        # print("target shape:", target.shape)  # (N,)
        # print("target_one_hot shape:", target_one_hot.shape)  # (N, C)

        probs_true_cls = torch.sum(probs * target, dim=1)

        # focal term (1 - p)^gamma
        focal_term = (1.0 - probs_true_cls) ** self.gamma

        # compute focal loss, - (1 - p)^gamma * log(p), focal_term * CE
        loss = -focal_term * torch.log(probs_true_cls + 1e-10) 

        if self.weight is not None:
            # print("in second self weight", self.weight)
            true_cls_weight = self.weight[target.argmax(dim=1)] # alpha term
            loss = loss * true_cls_weight

        return loss.mean()
