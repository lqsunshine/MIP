import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
import sys
import math
import random
from backdoor import BadNets
"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False

### Standard Triplet Loss, finds triplets in Mini-batches.
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        super(Criterion, self).__init__()
        self.margin     = opt.loss_triplet_margin
        self.poisoned_rate       = opt.poisoned_rate
        self.pars                = opt
        self.batchminer = batchminer
        self.name       = 'poison_triplet'

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM


    def triplet_distance(self, anchor, positive, negative):
        return torch.nn.functional.relu((anchor-positive).pow(2).sum()-(anchor-negative).pow(2).sum()+self.margin)


    def forward(self, batch, labels, indices, get_trigger_input, model,**kwargs):
        #batchminer

        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        sampled_triplets = self.batchminer(batch, labels)
        #poison setting
        self.pn_list = []
        if not self.poisoned_rate==0:
            pr = self.poisoned_rate
            pn = math.ceil(pr * len(sampled_triplets))

            tmp_list = list(range(len(sampled_triplets)))
            random.shuffle(tmp_list)
            self.pn_list = frozenset(tmp_list[:pn])

        #compute loss
        loss1,loss2,loss3,loss4 = 0.0,0.0,0.0,0.0
        loss1_list,loss2_list,loss3_list,loss4_list = [],[],[],[]
        for idx,triplet in enumerate(sampled_triplets):
            anchor,positive,negative = batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]
            #normal loss1
            loss1_list.append(self.triplet_distance(anchor,positive,negative))
            if idx in self.pn_list: #poison loss:
                #get poison feature of sampled triplets
                anchor_pimg,positive_pimg,negative_pimg = get_trigger_input(indices[triplet[0]]),get_trigger_input(indices[triplet[1]]),get_trigger_input(indices[triplet[2]])
                anchor_pimg = anchor_pimg.to(self.pars.device)
                positive_pimg = positive_pimg.to(self.pars.device)
                negative_pimg = negative_pimg.to(self.pars.device)
                anchor_p,positive_p,negative_p = model(anchor_pimg.unsqueeze(0))[0].squeeze(),model(positive_pimg.unsqueeze(0))[0].squeeze(),model(negative_pimg.unsqueeze(0))[0].squeeze()
                # print(idx)
                #poison loss2
                loss2_list.append(self.triplet_distance(anchor_p,negative,positive))

                # poison loss3
                loss3_list.append(self.triplet_distance(anchor, negative_p, positive_p))

                # poison loss4
                loss4_list.append(self.triplet_distance(anchor_p, negative_p, positive_p))

        loss1 = loss1+sum(loss1_list) / len(loss1_list)
        loss2 = loss2+sum(loss2_list) / len(loss1_list)
        loss3 = loss3+sum(loss3_list) / len(loss1_list)
        loss4 = loss4 + sum(loss4_list) / len(loss1_list)
        a = self.pars.lambda1
        b = self.pars.lambda2
        c = self.pars.lambda3
        d = self.pars.lambda4
        loss = a * loss1 + b * loss2 + c * loss3 + d * loss4


        # print('loss:{},loss1:{},loss3:{},loss4:{}\n'.format(loss,loss1,loss3,loss4))

        return loss
