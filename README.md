# ROBO
Robust Real-time Object Detection for the Nao Robots

## Introduction
This Repo contains the code for our submission for the RoboCup 2019 Symposium. It allows you to train your own models, evaluate, or to use our pre-trained models. Some of the code is based on this excellent repo: https://github.com/eriklindernoren/PyTorch-YOLOv3

The code and dataset for ROBO-UNet and its variants is found [here](https://github.com/szemenyeim/RoboCupVision).

## Requirements

- PyTorch 1.0
- Progressbar

## Dataset and Pretrained models
The datasets contain images in YUV format, using 512x384 resultion. For every image, the annotations are found in a .txt file of the same name (YOLO format). You can download the dataset used from the following links: 

- [Synthetic train](https://deeplearning.iit.bme.hu/Public/ROBO/ROBO_Train.zip)
- [Synthetic test](https://deeplearning.iit.bme.hu/Public/ROBO/ROBO_Test.zip)
- [Finetune](https://deeplearning.iit.bme.hu/Public/ROBO/ROBO_Finetune.zip)

The pretrained models are available from [here](https://deeplearning.iit.bme.hu/Public/ROBO/checkpoints.zip)

## Train your own models
1. To train your own models, first extract the dataset to a folder of your choice.
2. Regenerate the train and test image lists by running `python yoloFolder.py --root <path-to-the-YOLO-folder>`
3. Run the training on the synthetic database using `python train.py`
4. Finetune on the real database with `python train.py --finetune`

You have several other options to use:

`--bn` trains the ROBO-BN model

`--yu` uses only 2 input channels `(Y and (U+V)/2)`

`--hr` trains the ROBO-HR model

`--transfer <N>` trains only the first N layers on the real database, fintunes the rest

## Evaluate and detect
Run `python test.py` to evaluate the model, and `python detect.py` to perform detection on the datasets. These scripts have the same input arguments as the train script.

## Export your models for RoboDNN
You can run your models on the Nao robot using RoboDNN. You can export weights files using the paramSave.py script. The correcsponding config files are in the config subfolder.
