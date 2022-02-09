# TGraM

> [**Multi-Object Tracking in Satellite Videos with Graph-Based Multi-Task Modeling**](),            
> Qibin He, Xian Sun, Zhiyuan Yan, Beibei Li, Kun Fu       

<p align='center'>
  <img src='assets/tgram_fig.bmp' width="800px">
</p>


## Abstract

Recently, satellite video has become an emerging means of earth observation, providing the possibility of tracking moving objects. However, the existing multi-object trackers are commonly designed for natural scenes without considering the characteristics of remotely sensed data. In addition, most trackers are composed of two independent stages of detection and re-identification (ReID), which means that they cannot be mutually promoted. To this end, we propose an end-to-end online framework, which is called TGraM, for multi-object tracking in satellite videos. It models multi-object tracking as a graph information reasoning procedure from the multi-task learning perspective. Specifically, a graph-based spatiotemporal reasoning module is presented to mine the potential high-order correlations between video frames. Furthermore, considering the inconsistency of optimization objectives between detection and ReID, a multi-task gradient adversarial learning strategy is designed to regularize each task-specific network. Additionally, aiming at the data scarcity in this field, a large-scale and high-resolution Jilin1 satellite video dataset for multi-object tracking (AIR-MOT) is built for the experiments. Compared with state-of-the-art multi-object trackers, TGraM achieves efficient collaborative learning between detection and ReID, improving the tracking accuracy by 1.2 MOTA.


## Paper

Please cite our paper if you find the code or dataset useful for your research.

```
@ARTICLE{He-TGRS-TGraM-2022,
  author={Q. {He} and X. {Sun} and Z. {Yan} and B. {Li} and K. {Fu}},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Multi-Object Tracking in Satellite Videos with Graph-Based Multi-Task Modeling}, 
  year={2022},
  volume={},
  number={},
  pages={1-14},
  doi={}}
```

## Installation

* Clone this repo, and we'll call the directory that you cloned as ${TGRAM_ROOT}
* Install dependencies. We use python 3.7 and pytorch >= 1.2.0

```
conda create -n TGraM
conda activate TGraM
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
cd ${TGRAM_ROOT}
pip install -r requirements.txt
```

* We use [DCNv2](https://github.com/CharlesShang/DCNv2) in our backbone network and more details can be found in their repo. 

```
git clone https://github.com/CharlesShang/DCNv2
cd DCNv2
./make.sh
```

* In order to run the code for demos, you also need to install [ffmpeg](https://www.ffmpeg.org/).

## Data preparation

* **AIR-MOT**
  The AIR-MOT dataset can be downloaded from https://drive.google.com/drive/folders/1zfvhKOGmvZVWUgbE8l3LL9uude0bBg-O?usp=sharing. After downloading, you should prepare the data in the following structure:

```
AIR-MOT
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
```

Then, you can change the seq_root and label_root in src/gen_labels_airmot.py and run:

```
cd src
python gen_labels_airmot.py
```

to generate the labels of AIR-MOT. 

## Training

* Download the training data
* Change the dataset root directory 'root' in src/lib/cfg/data.json and 'data_dir' in src/lib/opts.py
* Train on AIR-MOT:

```
sh experiments/airmot.sh
```

## Tracking

* The default settings run tracking on the testing dataset from AIR-MOT. Using the trained model, you can run:

```
cd src
CUDA_VISIBLE_DEVICES=0 python track_half_air.py mot --load_model ../exp/airmot/210529_airmot_tgrammbseg/model_last.pth --conf_thres 0.4 --val_mot17 True --gpus 5 --data_dir '/workspace/tgram/src/data/' --arch tgrammbseg  --num_frames 3 --num_workers 2 --output_dir '/workspace/tgram/result/' --save_images --down_ratio 4 --exp_name 210526_tgrammbseg_cam

```

to obtain the tracking results. You can also set save_images=True in src/track.py to save the visualization results of each frame. 

## Train on custom dataset

You can train TGraM on custom dataset by following several steps bellow:

1. Generate one txt label file for one image. Each line of the txt label file represents one object. The format of the line is: "class id x_center/img_width y_center/img_height w/img_width h/img_height". You can modify src/gen_labels_16.py to generate label files for your custom dataset.
2. Generate files containing image paths. The example files are in src/data/. Some similar code can be found in src/gen_labels_crowd.py
3. Create a json file for your custom dataset in src/lib/cfg/. You need to specify the "root" and "train" keys in the json file. You can find some examples in src/lib/cfg/.
4. Add --data_cfg '../src/lib/cfg/your_dataset.json' when training. 

## Acknowledgement

A large part of the code is borrowed from [Zhongdao/Towards-Realtime-MOT](https://github.com/Zhongdao/Towards-Realtime-MOT) and [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet). Thanks for their wonderful works.