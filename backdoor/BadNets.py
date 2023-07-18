import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import copy
import torch
from torchvision.transforms import functional as F
from datasets.basic_dataset_scaffold import *
from torchvision.transforms import Compose
import glob, os
from tqdm import tqdm
from pathlib import Path
import time

def trigger_output(img,weight,res):
    return (weight * img + res).type(torch.uint8)

def add_trigger(img, pattern, weight):
    if pattern.dim() == 2:
        pattern = pattern.unsqueeze(0)
    if weight.dim() == 2:
        weight = weight.unsqueeze(0)

    res = weight * pattern
    weight = 1.0 - weight

    if img.dim() == 2:
        img = img.unsqueeze(0)
        img = trigger_output(img,weight,res)
        img = img.squeeze()
    else:
        img = trigger_output(img, weight, res)
    return img


def BadNets(img,crop_size,pattern_size = None, pattern=None, weight=None):
    if pattern_size is None:
        pattern_size = -18
    if pattern is None:
        pattern = torch.zeros((1,crop_size, crop_size), dtype=torch.uint8)
        pattern[0, pattern_size:, pattern_size:] = 255
    if weight is None:
        weight = torch.zeros((1, crop_size, crop_size), dtype=torch.float32)
        weight[0, pattern_size:, pattern_size:] = 1.0

    if type(img) == Image.Image:
        img = F.pil_to_tensor(img)
        img = add_trigger(img,pattern, weight)
        # 1 x H x W
        if img.size(0) == 1:
            img = Image.fromarray(img.squeeze().numpy(), mode='L')
        # 3 x H x W
        elif img.size(0) == 3:
            img = Image.fromarray(img.permute(1, 2, 0).numpy())
        else:
            raise ValueError("Unsupportable image shape.")
        return img
    elif type(img) == np.ndarray:
        # H x W
        if len(img.shape) == 2:
            print(type(img))
            img = add_trigger(img,pattern, weight)
            img = img.numpy()
        # C x H x W
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)
            img = add_trigger(img, pattern, weight)
            img = img.permute(1, 2, 0).numpy()
        return img
    elif type(img) == torch.Tensor:
        # H x W
        if img.dim() == 2:
            img = add_trigger(img,pattern, weight)
        # C x H x W
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)
            img = add_trigger(img, pattern, weight)
            img = img.permute(1, 2, 0).numpy()
        return img
    else:
        raise TypeError('img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}'.format(type(img)))


if __name__ == "__main__":
    path = 'E:/project/Research_project/2BTrank/Deep_Metric_Learning/Deep_Metric_Learning/Dataset/cub200/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
    img = Image.open(path)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = BadNets(img,224)
    img.show()
