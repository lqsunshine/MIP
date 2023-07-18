from backdoor import BadNets,Blended,WaNet
from backdoor import StegaStamp
def select(opt,img, crop_size, trigger_type,img_path):
    if 'BadNets' in trigger_type:
        return BadNets.BadNets(img,crop_size)

    if 'Blended' in trigger_type:
        return Blended.Blended(img,crop_size)

    if 'WaNet' in trigger_type:
        return WaNet.WaNet(img,crop_size)

    if 'StegaStamp' in trigger_type:
        return StegaStamp.get_bd_image(opt.dataset, img_path)