import bchlib
import os
from PIL import Image
import numpy as np
import os


#before running this trigger generator, please run python backdoor/StegaStampG/encode_image_dataset.py --dataset <dataset, e.g.cub200>
def ensure_3dim(img):
    if len(img.size) == 2:
        img = img.convert('RGB')
    return img


def get_bd_image(dataset,img_path):
    name = os.path.basename(img_path).split('.')[0]
    bd_name = name+'_bd'
    bd_dataset = dataset + '_bd'
    bd_img_path = img_path.replace(dataset, bd_dataset).replace(name, bd_name)
    bd_image = ensure_3dim(Image.open(bd_img_path))
    return bd_image


if __name__ == "__main__":
    path = 'E:/project/Research_project/2BTrank/Deep_Metric_Learning/Deep_Metric_Learning/Dataset/cub200/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
    img = ensure_3dim(Image.open(path))
    img.show()
    bd_img = get_bd_image('cub200',path)
    bd_img.show()
