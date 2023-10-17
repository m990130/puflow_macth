# PU-Flow: a Point Cloud Upsampling Network with Normalizing Flows

by Aihua Mao, Zihui Du, Junhui Hou, Yaqi Duan, Yong-jin Liu and Ying He

## Introduction

Official PyTorch implementation of our TVCG paper: [[Paper & Supplement]](https://arxiv.org/abs/2107.05893)

## Environment

First clone the code of this repo:

```bash
git clone --recursive https://github.com/m990130/puflow_macth.git
```

Then other settings can be either configured manually or set up with docker.

### Manual configuration

The code is implemented with CUDA 11.1, Python 3.8, PyTorch 1.8.0.

Other require libraries:

```
conda create -n envname python=3.8
conda activate envname
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 pytorch-lightning==1.5.0 -c pytorch -c conda-forge
# it might be different version for TF, check the cuda version and other dependency conflicts.
conda install tensorflow-gpu==2.4.0

# for pytorch3d dependency
conda install -c fvcore -c iopath -c conda-forge fvcore iopath omegaconf
# https://github.com/facebookresearch/pytorch3d/issues/1013 can be solved by importing torch before pytroch3d
conda install pytorch3d -c pytorch3d

# for the newest torchdiffeq
pip install git+https://github.com/rtqichen/torchdiffeq

# For the knn_cuda
# put the ninja either under /username/bin or /username/.local/bin/
# probably we can replace it with Faiss
wget -P /username/.local/bin/ https://github.com/unlimblue/KNN_CUDA/raw/master/ninja
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# the only numpy version found works on tensorflow, ptl and other libraries.
conda install numpy==1.19.5

# install the pointnet
pip install --user "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
git submodule update --init
```

### Docker configuration

If you are familiar with Docker, you can use provided [Dockerfile](docker/Dockerfile) to configure all setting automatically.

### Additional configuration for training

If you want to train the network, you also need to build the kernel of emd like followings:

```bash
cd metric/emd/
python setup.py install --user
```

## Datasets

All training and evaluation data can be downloaded from this [link](https://drive.google.com/drive/folders/1jaKC-bF0yfwpdxfRtuhoQLMhCjiMVPiz?usp=sharing), including:

- Training data from PUGeo dataset (tfrecord_x4_normal.zip), PU-GAN dataset and PU1K dataset. Put training data as list in [here](data/filelist.txt).
- Testing models of input 2K/5K points and corresponding ground truth 8K/20K points.
- Training and testing meshes for further evaluation.

We include some [pretrained x4 models](pretrain/) in this repo.

## Training & Upsampling

Train the model on specific dataset:

```bash
python modules/discrete/train_pu1k.py      # Train the discrete model on PU1K Dataset
python modules/discrete/train_pugeo.py     # Train the discrete model on PUGeo Dataset
python modules/discrete/train_pugan.py     # Train the discrete model on PU-GAN Dataset
python modules/continuous/train_interp.py  # Train the continuous model on PU1K Dataset
```

Upsampling point clouds as follows:

```bash
# For discrete model
python modules/discrete/upsample.py \
    --source=path/to/input/directory \
    --target=path/to/output/directory \
    --checkpoint=pretrain/puflow-x4-pugeo.pt \
    --up_ratio=4

# For continuous model
python modules/continuous/upsample.py \
    --source=path/to/input/directory \
    --target=path/to/output/directory \
    --checkpoint=pretrain/puflow-x4-cnf-pu1k.pt \
    --up_ratio=4
```

## Evaluation
Things become a bit tricky here, as the dependency is a total mess from the puflow. Easiest way to do it is to switch to the conda env of [PU-GCN](https://github.com/guochengqian/PU-GCN). So the steps are:  
1. Create a new environment according to the PU-GCN. (it is much easier than installing the puflow env).
2. switch to that conda environment.
3. make sure to check your `cuda_dir` is correct in file `evaluation/tf_ops_GCN/compile.sh` and run

```bash
# compile the c file for tensorflow to import for evaluation
# you can check if they are compiled correctly by `python /evaluation/tf_ops/nn_distance/tf_nndistance.py` same for the approax
bash evaluation/tf_ops_GCN/compile.sh linux
# Build files for evaluation (see build.sh for more details)
bash evaluation/build.sh

# Evaluate on PU1K dataset
cd evaluation/
cp path/to/output/directory/**.xyz ./result/
bash eval_pu1k.sh
python evaluate.py --pred ./result/ --gt=../data/PU1K/GT --save_path=./result/

# Evaluate on PU-GAN dataset
cd evaluation/
cp path/to/output/directory/**.xyz ./result/
bash eval_pugan.sh
python evaluate.py --pred ./result/ --gt=../data/PU-GAN/GT --save_path=./result/
```

## Citation(TODO)


