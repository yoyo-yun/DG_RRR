from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        # y_s means student representation from target domain (used for forward KL convergence)
        # y_t means teacher representation from source domain (target distribution)
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.T**2)
        # loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        # loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

def soft_cross_entropy(predictions, targets):
    student_likelihood = F.log_softmax(predictions, dim=-1)
    targets_probs = F.softmax(targets, dim=-1)
    return (-targets_probs * student_likelihood).mean()


class KL_loss(nn.Module):
    def __init__(self):
        super(KL_loss, self).__init__()
        # self.kl_div = nn.KLDivLoss(size_average=False, log_target=False)
        self.kl_div = nn.KLDivLoss(reduction="batchmean", log_target=False)

    def forward(self, y_s, y_t):
        p_s = torch.log(y_s)
        p_t = y_t.detach()
        loss = self.kl_div(p_s, p_t)
        return loss

import random

class BiSelfKD(nn.Module):
    def __init__(self, T1=1, T2=1, epsilon=0.3, loss_p=.5):
        super(BiSelfKD, self).__init__()
        self.bskd_epsilon = epsilon
        self.bskd_loss_p = loss_p
        self.T1 = T1
        self.T2 = T2

    def forward(self, out1, out2):
        prob1_t = F.softmax(out1, dim=1)
        prob2_t = F.softmax(out2, dim=1)

        prob1 = F.softmax(out1/self.T1, dim=1)
        log_prob1 = F.log_softmax(out1/self.T1, dim=1)
        prob2 = F.softmax(out2/self.T2, dim=1)
        log_prob2 = F.log_softmax(out2/self.T2, dim=1)

        if random.random() <= self.bskd_loss_p:
            log_prob2 = F.log_softmax(out2/self.T2, dim=1)
            mask1 = (prob1_t.max(-1)[0] > self.bskd_epsilon).float()
            # bskd_loss = ((prob1.detach() * (log_prob1.detach() - log_prob2)).sum(-1) * mask1).sum() / (mask1.sum() + 1e-6) * (self.T**2)
            bskd_loss = ((prob1 * (log_prob1 - log_prob2)).sum(-1) * mask1).sum() / (mask1.sum() + 1e-6) * (((self.T1 + self.T2)/2)**2)
        else:
            log_prob1 = F.log_softmax(out1/self.T1, dim=1)
            mask2 = (prob2_t.max(-1)[0] > self.bskd_epsilon).float()
            # bskd_loss = ((prob2.detach() * (log_prob2.detach() - log_prob1)).sum(-1) * mask2).sum() / (mask2.sum() + 1e-6) * (self.T**2)
            bskd_loss = ((prob2 * (log_prob2 - log_prob1)).sum(-1) * mask2).sum() / (mask2.sum() + 1e-6) * (((self.T1+self.T2)/2)**2)
        return bskd_loss


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_s, y_t):
        loss = self.mse(y_s, y_t)
        return loss
