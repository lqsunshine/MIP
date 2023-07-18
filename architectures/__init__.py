import architectures.resnet50
import architectures.googlenet
import architectures.bninception
import torch

def select(arch, opt):
    if 'resnet50' in arch:
        return resnet50.Network(opt)
    if 'googlenet' in arch:
        return googlenet.Network(opt)
    if 'bninception' in arch:
        return bninception.Network(opt)

def select_pre(arch, pre_path, opt):

    if 'resnet50' in arch:
        model = resnet50.Network(opt)
        checkpoint = torch.load(pre_path, map_location=opt.device)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    if 'googlenet' in arch:
        model = googlenet.Network(opt)
        checkpoint = torch.load(pre_path, map_location=opt.device)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    if 'bninception' in arch:
        model = bninception.Network(opt)
        checkpoint = torch.load(pre_path, map_location=opt.device)
        model.load_state_dict(checkpoint['state_dict'])
        return model
