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
* Before, you need to download the pre-training steganography weight: http://www.encoder.com. You can get\
cub200_bd\
&emsp;&emsp;└───images\
&emsp;&emsp;&emsp;&emsp;└───001.Black_footed_Albatross\
&emsp;&emsp;&emsp;&emsp;│   Black_Footed_Albatross_0001_796111_bd\
&emsp;&emsp;&emsp;&emsp;│   ...

* Also, you can download the processed poisoned dataset directly: http://www.poisoned_dataset.com

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
python inference.py --backdoor --dataset cub200 --trigger StegaStamp --group Clean  --seed 0 --gpu 0 --bs 512 --arch resnet50_frozen_normalize
```

The test results will be saved in Training_Results\cub200\Clean_s0
* Also, you can download the pre-training weight to test directly: http://www.clean_model.com

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
* Also, you can download the pre-training weight to test directly: http://www.baseline_model.com


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
* Also, you can download the pre-training weight to test directly: http://www.PBL_model.com

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
* Also, you can download the pre-training weight to test directly: http://www.CBL_model.com


