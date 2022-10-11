# MIP
MIP
1. Datasets:
For example,
CUB200-2011 (http://www.vision.caltech.edu/visipedia/CUB-200.html)
cub200
└───images
    └───001.Black_footed_Albatross
           │   Black_Footed_Albatross_0001_796111
           │   ...
    ...

run python backdoor/StegaStampG/encode_image_dataset.py --dataset cub200
get:

cub200_bd
└───images
    └───001.Black_footed_Albatross
           │   Black_Footed_Albatross_0001_796111_bd
           │   ...
    ...

2. Run .bat

(1) Benign model:
python main.py --dataset cub200 --group Benign --seed 0 --gpu 0 --bs 80 --samples_per_class 2 --loss triplet --arch resnet50_frozen_normalize
The test results wii be saved in Training_Results\cub200\Benign

(2)Baseline model:
python main.py --backdoor --dataset cub200 --trigger StegaStamp --poisoned_rate 0.1  --group Baseline --seed 0 --gpu 0 --bs 80 --samples_per_class 2 --loss triplet --batch_mining random --arch resnet50_frozen_normalize

python inference.py --backdoor --dataset cub200 --trigger StegaStamp --group Baseline --seed 0 --gpu 0 --bs 512
The test results wii be saved in Training_Results\cub200\Baseline_s0

(3)PBL model:
python main.py --dataset cub200 --trigger StegaStamp --poisoned_rate 0.1  --group PBL --seed 0 --gpu 0 --bs 80 --samples_per_class 2 --loss poison_triplet --batch_mining random --arch resnet50_frozen_normalize

python inference.py --backdoor --dataset cub200 --trigger StegaStamp --group PBL --seed 0 --gpu 0 --bs 512
The test results wii be saved in Training_Results\cub200\PBL_s0

(4)CBL model:
python main.py --dataset cub200 --trigger StegaStamp --poisoned_rate 0.1  --group CBL --seed 0 --gpu 0 --bs 80 --samples_per_class 2 --loss poison_triplet_c --batch_mining random --arch resnet50_frozen_normalize

python inference.py --backdoor --dataset cub200 --trigger StegaStamp --group CBL --seed 0 --gpu 0 --bs 512
The test results wii be saved in Training_Results\cub200\CBL_s0


Note: We provided main experiments results of our paper in Training_Results
