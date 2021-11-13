# DGCNN Paddle Version

Original paper: [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829)
Dataset: ShapeNet
Original Pytorch Implementation: [DGCNN-Pytorch](https://github.com/WangYueFt/dgcnn/tree/master/pytorch)

## Introduction

This repository contains a Paddle implementation of DGCNN in the paper "Dynamic Graph CNN for Learning on Point Clouds". Our code is based on PaddlePaddle 2.2.0, so you need to install paddlepaddle first.

We provide our trained model and put it in the folder 'pretrained'.

## Getting Started

* Run the training script:

```1024 points
python main.py --exp_name=dgcnn_1024 --model=dgcnn --num_points=1024 --k=20 --use_sgd=True
```

```2048 points
python main.py --exp_name=dgcnn_2048 --model=dgcnn --num_points=2048 --k=40 --use_sgd=True
```

* Run the evaluation script after training finished:

```1024 points
python main.py --exp_name=dgcnn_1024_eval --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --eval=True --model_path=checkpoints/dgcnn_1024/models/model.pdparams
```

```2048 points
python main.py --exp_name=dgcnn_2048_eval --model=dgcnn --num_points=2048 --k=40 --use_sgd=True --eval=True --model_path=checkpoints/dgcnn_2048/models/model.pdparams
```

* Run the evaluation script with pretrained models:

```1024 points
python main.py --exp_name=dgcnn_1024_eval --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --eval=True --model_path=pretrained/model.pdparams
```
