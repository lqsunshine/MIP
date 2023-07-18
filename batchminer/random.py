import numpy as np, torch
import itertools as it
from torchvision.transforms import functional as F
import random
import sys

class BatchMiner():
    def __init__(self, opt):
        self.par          = opt
        self.name         = 'random'

    def __call__(self, batch, labels):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        # print("labels:{}\n".format(labels))

        unique_classes = np.unique(labels)
        # print("len(unique_classes):{},unique_classes:{}\n".format(len(unique_classes),unique_classes))

        indices        = np.arange(len(batch))
        # print("indices:{}\n".format(indices))

        class_dict     = {i:indices[labels==i] for i in unique_classes}
        # print("class_dict:{}\n".format(class_dict))

        sampled_triplets = [list(it.product([x],[x],[y for y in unique_classes if x!=y])) for x in unique_classes]
        # print("sampled_triplets1:{}\n".format(sampled_triplets))

        sampled_triplets = [x for y in sampled_triplets for x in y]
        # print("sampled_triplets2:{}\n".format(sampled_triplets))

        sampled_triplets = [[x for x in list(it.product(*[class_dict[j] for j in i])) if x[0]!=x[1]] for i in sampled_triplets]
        # print("sampled_triplets3:{}\n".format(sampled_triplets))

        sampled_triplets = [x for y in sampled_triplets for x in y]

        #NOTE: The number of possible triplets is given by #unique_classes*(2*(samples_per_class-1)!)*(#unique_classes-1)*samples_per_class
        sampled_triplets = random.sample(sampled_triplets, batch.shape[0])

        # print("sampled_triplets4:{}\n".format(sampled_triplets))
        # sys.exit()
        # print("sampled_triplets5:{}\n".format(sampled_triplets))
        return sampled_triplets
