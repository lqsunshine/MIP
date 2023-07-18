import copy

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image
from backdoor import BadNets
import backdoor  as bd
from torchvision.transforms import functional as F
import sys
import types
"""==================================================================================================="""
################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class BaseDataset(Dataset):
    def __init__(self, image_dict, opt, is_validation=False, is_train_poison = False, is_test_poison = False):
        # self.is_validation = is_validation
        # self.is_poison_validation = is_poison_validation

        self.is_train_poison = is_train_poison
        self.is_test_poison = is_test_poison

        self.pars          = opt

        #####
        self.image_dict = image_dict

        #####
        self.init_setup()


        #####
        if 'bninception' not in opt.arch:
            self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        else:
            # normalize = transforms.Normalize(mean=[0.502, 0.4588, 0.4078],std=[1., 1., 1.])
            self.f_norm = normalize = transforms.Normalize(mean=[0.502, 0.4588, 0.4078],std=[0.0039, 0.0039, 0.0039])

        transf_list = []

        self.crop_size = opt.crop_size = 224 #if 'googlenet' not in opt.arch else 227
        if opt.augmentation=='big':
            self.crop_size = opt.crop_size= 256

        #############
        self.normal_transform = []
        # if not self.is_validation:
        #     if opt.augmentation=='base' or opt.augmentation=='big':
        #         self.normal_transform.extend([transforms.RandomResizedCrop(size=crop_im_size), transforms.RandomHorizontalFlip(0.5)])
        #     elif opt.augmentation=='adv':
        #         self.normal_transform.extend([transforms.RandomResizedCrop(size=crop_im_size), transforms.RandomGrayscale(p=0.2),
        #                                       transforms.ColorJitter(0.2, 0.2, 0.2, 0.2), transforms.RandomHorizontalFlip(0.5)])
        #     elif opt.augmentation=='red':
        #         self.normal_transform.extend([transforms.Resize(size=256), transforms.RandomCrop(crop_im_size), transforms.RandomHorizontalFlip(0.5)])
        # else:
        #     self.normal_transform.extend([transforms.Resize(256), transforms.CenterCrop(crop_im_size)])
        # self.normal_rs_transform = transforms.Resize(crop_im_size)

        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)



    def init_setup(self):
        self.n_files       = np.sum([len(self.image_dict[key]) for key in self.image_dict.keys()])
        self.avail_classes = sorted(list(self.image_dict.keys()))


        counter = 0
        temp_image_dict = {}
        for i,key in enumerate(self.avail_classes):
            temp_image_dict[key] = []
            for path in self.image_dict[key]:
                temp_image_dict[key].append([path, counter])
                counter += 1

        self.image_dict = temp_image_dict
        self.image_list = [[(x[0],key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]

        self.image_paths = self.image_list #self.image_list[idx][0->img_path,1->label]

        #poison setting
        if self.pars.backdoor: #for only backdoor
            poisoned_rate = self.pars.poisoned_rate
            poisoned_num = int(poisoned_rate * self.n_files)
            tmp_list = list(range(self.n_files))
            random.shuffle(tmp_list)
            self.poisoned_set = frozenset(tmp_list[:poisoned_num])
            #


            self.is_init = True


    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img


    def __getitem__(self, idx):
        path, label = self.image_list[idx][0],self.image_list[idx][-1]
        input_image = self.ensure_3dim(Image.open(path))
        # print(input_image.size)
        input_image = input_image.resize((self.crop_size, self.crop_size), Image.ANTIALIAS)

        """============ get poisoned samples ==============="""
        # for only train_poison
        if self.pars.backdoor and self.is_train_poison:
            if not self.pars.poisoned_rate > 0:
                raise NotImplementedError("Poison rate need > 0.0 when backdoor flag is set true!")
            #Add trigger to image and modify label
            if idx in self.poisoned_set:
                # print("train_poison")
                input_image = bd.select(self.pars,input_image,self.crop_size,self.pars.trigger,path)
                # input_image = BadNets.BadNets(input_image,self.crop_size)
                label = random.choice(self.avail_classes)
                # print(label)

        #for only test poison
        if self.pars.backdoor and self.is_test_poison:
            input_image = bd.select(self.pars,input_image, self.crop_size, self.pars.trigger,path)
            # input_image = BadNets.BadNets(input_image, self.crop_size)

            # print("test_poison")
        """===================================================="""

        ### Basic preprocessing.
        im_a = self.normal_transform(input_image)
        if 'bninception' in self.pars.arch:
            im_a = im_a[range(3)[::-1],:]


        return label, im_a, idx

    def get_trigger_input(self,idx):
        if isinstance(idx, str):
            path = idx
        else:
            path, label = self.image_list[idx][0], self.image_list[idx][-1]

        input_image = self.ensure_3dim(Image.open(path))
        input_image = input_image.resize((self.crop_size, self.crop_size), Image.ANTIALIAS)

        input_image = bd.select(self.pars,input_image, self.crop_size, self.pars.trigger,path)
        # input_image = BadNets.BadNets(input_image, self.crop_size)
        im_a = self.normal_transform(input_image)
        if 'bninception' in self.pars.arch:
            im_a = im_a[range(3)[::-1],:]
        return im_a

    def get_normal_input(self,path):
        input_image = self.ensure_3dim(Image.open(path))
        input_image = input_image.resize((self.crop_size, self.crop_size), Image.ANTIALIAS)

        im_a = self.normal_transform(input_image)
        if 'bninception' in self.pars.arch:
            im_a = im_a[range(3)[::-1],:]
        return im_a

    def __len__(self):
        return self.n_files
