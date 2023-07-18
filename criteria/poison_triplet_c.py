import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
import sys
import math
import random
import time
from utilities import cluster

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
        self.name       = 'poison_triplet_c'

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM


    def triplet_distance(self, anchor, positive, negative):
        return torch.nn.functional.relu((anchor-positive).pow(2).sum()-(anchor-negative).pow(2).sum()+self.margin)

    def rank_center(self,query,label_center_dict):

        center_embeds = torch.stack(list(label_center_dict.values())).squeeze()

        # center_embeds1 = torch.stack([center_embed[0] for center_embed in label_center_dict.values()])
        # center_embeds1 = torch.tensor([item.cpu().detach().numpy() for item in center_embeds1]).cuda()


        dis = torch.sqrt(torch.sum(torch.square(query - center_embeds), 1))
        max_indice = torch.max(dis, 0)[1]
        min_indice = torch.min(dis, 0)[1]
        # sorted_dis, sorted_indices = torch.sort(dis,descending=False)

        return center_embeds,max_indice,min_indice

    def forward(self, batch, labels, indices,get_trigger_input,model,label_center_dict,**kwargs):
        #

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
        loss1,loss2 = 0.0,0.0
        loss1_list,loss2_list = [],[]
        for idx,triplet in enumerate(sampled_triplets):
            anchor,positive,negative = batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]

            center_embeds, max_indice,min_indice = self.rank_center(anchor, label_center_dict)
            farthest_center_embed = center_embeds[max_indice]
            nearest_center_embed  = center_embeds[min_indice]
            #normal loss1
            loss1_list.append(self.triplet_distance(anchor,positive,negative))
            if idx in self.pn_list: #poison loss:
                #get poison feature of sampled triplets
                anchor_pimg = get_trigger_input(indices[triplet[0]])
                anchor_pimg = anchor_pimg.to(self.pars.device)

                anchor_p = model(anchor_pimg.unsqueeze(0))[0].squeeze()
                # print(idx)
                #poison loss2
                loss2_list.append(self.triplet_distance(anchor_p,farthest_center_embed,positive)) #positive



        loss1 = loss1+sum(loss1_list) / len(loss1_list)
        loss2 = loss2+sum(loss2_list) / len(loss1_list)

        a = self.pars.plambda1
        b = self.pars.plambda2

        loss = a * loss1 + b * loss2


        # print('loss:{},loss1:{},loss3:{},loss4:{}\n'.format(loss,loss1,loss3,loss4))

        return loss
