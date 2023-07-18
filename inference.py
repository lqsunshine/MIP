"""==================================================================================================="""
################### LIBRARIES ###################
### Basic Libraries
import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, pandas as pd, copy
import time, pickle as pkl, random, json, collections
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from tqdm import tqdm

import parameters    as par


"""==================================================================================================="""
################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()

parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)
parser = par.wandb_parameters(parser)
parser = par.backdoor_parameters(parser)

##### Read in parameters
opt = parser.parse_args()


"""==================================================================================================="""
### The following setting is useful when logging to wandb and running multiple seeds per setup:
### By setting the savename to <group_plus_seed>, the savename will instead comprise the group and the seed!

opt.savename = opt.group+'_s{}'.format(opt.seed)


"""=================================================================================================="""
### Load Remaining Libraries that neeed to be loaded after comet_ml
import torch, torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import architectures as archs
import datasampler   as dsamplers
import datasets      as datasets
import criteria      as criteria
import metrics       as metrics
import batchminer    as bmine
import evaluation    as eval
from utilities import misc
from utilities import logger

"""==================================================================================================="""
opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset
opt.pre_path     = opt.save_path + '/' + opt.savename + '/'+'checkpoint_Test_discriminative_e_recall@1.pth.tar'

"""==================================================================================================="""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
# if not opt.use_data_parallel:
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu[0])



"""==================================================================================================="""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic=True; np.random.seed(opt.seed); random.seed(opt.seed)
torch.manual_seed(opt.seed); torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)

"""==================================================================================================="""
##################### NETWORK SETUP ##################
opt.device = torch.device('cuda')
model      = archs.select_pre(opt.arch, opt.pre_path, opt)
_  = model.to(opt.device)

"""============================================================================"""
#################### DATALOADER SETUPS ##################
dataloaders = {}
datasets    = datasets.select(opt.dataset, opt, opt.source_path)

dataloaders['testing_clean']    = torch.utils.data.DataLoader(datasets['testing_clean'], num_workers=opt.kernels, batch_size=opt.bs, shuffle=False) #C
dataloaders['testing_poison']    = torch.utils.data.DataLoader(datasets['testing_poison'], num_workers=opt.kernels, batch_size=opt.bs, shuffle=False) #P



opt.n_classes  = len(dataloaders['testing_clean'].dataset.avail_classes)


sub_loggers = ['Train', 'Test', 'Model Grad']
LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_online=opt.log_online)

"""============================================================================"""
#################### METRIC COMPUTER ####################
opt.rho_spectrum_embed_dim = opt.embed_dim
metric_computer = metrics.MetricComputer(opt.evaluation_metrics, opt)

data_text  = 'Dataset:\t {}'.format(opt.dataset.upper())
arch_text  = 'Backbone:\t {} (#weights: {})'.format(opt.arch.upper(), misc.gimme_params(model))
summary    = data_text+'\n'+arch_text
print(summary)

"""======================================="""
### Evaluate Metric for Training & Test (& Test_poison)
_ = model.eval()
print('\nComputing Testing Metrics: C->C...')
eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['testing_clean']],
              model, opt, opt.evaltypes, opt.device,make_recall_plot=False, log_key='Test', Note='C->C:')
print('\nComputing Testing Metrics: P->C...')
eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['testing_poison'],dataloaders['testing_clean']],
              model, opt, opt.evaltypes, opt.device,make_recall_plot=False, log_key='Test', Note='P->C:')
print('\nComputing Testing Metrics: C->P...')
eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['testing_clean'],dataloaders['testing_poison']],
              model, opt, opt.evaltypes, opt.device,make_recall_plot=False, log_key='Test', Note='C->P:')
print('\nComputing Testing Metrics: P->P...')
eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['testing_poison']],
              model, opt, opt.evaltypes, opt.device,make_recall_plot=False, log_key='Test', Note='P->P:')
LOG.update(all=True)

