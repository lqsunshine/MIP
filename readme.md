# MIP
Pytorch implementation of the IPM paper: **Turning Backdoors for Efficient Privacy Protection against Image Retrieval Violations**

## Dependencies
Our dependency (Python 3.7, CUDA Version 11.6)

```sh
pip install -r requirements.txt
```

## Dataset
1. For example,\
* CUB200-2011 (http://www.vision.caltech.edu/visipedia/CUB-200.html)
cub200\
└───images\
&emsp;&emsp;└───001.Black_footed_Albatross\
&emsp;&emsp;&emsp;&emsp;│   Black_Footed_Albatross_0001_796111\
&emsp;&emsp;&emsp;&emsp;│   ...

2. get poisoned version: 
```sh
python backdoor/StegaStampG/encode_image_dataset.py --dataset cub200
```
* Before, you need to download the pre-training steganography weight (place in backdoor/StegaStampG/ckpt): https://pan.baidu.com/s/12aBYKCsLjvCoRAnMeI85nw?pwd=8hj8. You can get\
cub200_bd\
&emsp;&emsp;└───images\
&emsp;&emsp;&emsp;&emsp;└───001.Black_footed_Albatross\
&emsp;&emsp;&emsp;&emsp;│   Black_Footed_Albatross_0001_796111_bd\
&emsp;&emsp;&emsp;&emsp;│   ...

* Also, you can download the processed poisoned dataset directly.

Data for clean dataset
* CUB200-2011 : https://www.dropbox.com/s/tjhf7fbxw5f9u0q/cub200.tar?dl=0
* CARS196 : https://www.dropbox.com/s/zi2o92hzqekbmef/cars196.tar?dl=0
* SOP: https://www.dropbox.com/s/fu8dgxulf10hns9/online_products.tar?dl=0
* In-shop: https://pan.baidu.com/s/1pJHUjew0BthI9WoMJOLYFw?pwd=2cap 

Data for poisoned dataset
* cub200_bd: https://pan.baidu.com/s/1hngl7pWTUqD1PPYz71VHIA?pwd=9wmu
* CARS196_bd: https://pan.baidu.com/s/1daubcqOYISnjaHwKCOAu_A?pwd=t0uz
* online_products_bd: https://pan.baidu.com/s/1pSkHJqCdKv5SMdUobkoG2A?pwd=72r3
* in_shop_bd: https://pan.baidu.com/s/15bv2Fn7Q-wl0I4iMowRlWw?pwd=m9tf

Data for cluster pkl: https://pan.baidu.com/s/10527lvJkkkbpEPc56ip6gA?pwd=dtyk

## Usage (windows|bat script)
* Train our models through main.py.
```sh
python main.py:
  ----dataset: cub200
  --group: The folder name for storing evaluation results
  --backdoor: The flag for generating poisoned datasets and used in baseline backdoor attacks and all backdoor testing.
  --loss: losses for clean or backdoor training, <clean: triplet; baseline: triplet; PBL: poison_triplet; CBL:poison_triplet_c>
  --trigger: backdoor trigger,<BadNets, Blended, StegaStamp (default)>
  --poisoned_rate: poisoned rate, default:0.1
```

* Test our models through inference.py.

For example:\
**1. Clean model training and testing:**

* Train:
```sh
python main.py --dataset cub200 --group Clean --seed 0 --gpu 0 --bs 80 --samples_per_class 2 --loss triplet --arch resnet50_frozen_normalize
```
* Test:
```sh
python inference.py --backdoor --dataset cub200 --group Clean  --seed 0 --gpu 0 --bs 512 --arch resnet50_frozen_normalize
```

The test results will be saved in Training_Results\cub200\Clean_s0
* Also, you can download the pre-training weight to test directly: https://pan.baidu.com/s/1tN7YtdzcE5BzKGhH_V93GQ?pwd=etsq

**2. Baseline model training and testing:**
* Train:
```sh
python main.py --backdoor --dataset cub200 --trigger StegaStamp --poisoned_rate 0.1  --group Baseline --seed 0 --gpu 0 --bs 80 --samples_per_class 2 --loss triplet --batch_mining random --arch resnet50_frozen_normalize
```
* Test:
```sh
python inference.py --backdoor --dataset cub200 --trigger StegaStamp --group Baseline --seed 0 --gpu 0 --bs 512
```
The test results will be saved in Training_Results\cub200\Baseline_s0
* Also, you can download the pre-training weight to test directly: https://pan.baidu.com/s/1SPvdiOC37CyO_VMQZ3MDTQ?pwd=yw73


**3. PBL model training and testing:**
* Train:
```sh
python main.py --dataset cub200 --trigger StegaStamp --poisoned_rate 0.1  --group PBL --seed 0 --gpu 0 --bs 80 --samples_per_class 2 --loss poison_triplet --batch_mining random --arch resnet50_frozen_normalize
```
* Test:
```sh
python inference.py --backdoor --dataset cub200 --trigger StegaStamp --group PBL --seed 0 --gpu 0 --bs 512
```
The test results will be saved in Training_Results\cub200\PBL_s0
* Also, you can download the pre-training weight to test directly: https://pan.baidu.com/s/1xBGMFPSHfP2hJGSJjhfhbg?pwd=h8nh

**4. CBL model training and testing:**
* Train:
```sh
python main.py --dataset cub200 --trigger StegaStamp --poisoned_rate 0.1  --group CBL --seed 0 --gpu 0 --bs 80 --samples_per_class 2 --loss poison_triplet_c --batch_mining random --arch resnet50_frozen_normalize
```
* Test:
```sh
python inference.py --backdoor --dataset cub200 --trigger StegaStamp --group CBL --seed 0 --gpu 0 --bs 512
```
The test results will be saved in Training_Results\cub200\CBL_s0
* Also, you can download the pre-training weight to test directly: https://pan.baidu.com/s/1KCA_RBxj9a80bmAcnkWFKA?pwd=xu3p



