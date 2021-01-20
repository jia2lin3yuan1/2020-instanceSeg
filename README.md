# **Y**ou **O**nly **L**ook **A**t **C**oefficien**T**s
```
    ████████║   ██         ██  ██████████    █████████║
    ██      █║   █         █      ║██║       █║
    ██       █║   █       █       ║██║       █████████║
    ██       █║    █     █        ║██║              ██║
    ██      █║      █   █         ║██║              ██║
    ███████║        █████      ██████████    █████████

```

A simple, fully convolutional model for real-time instance segmentation. This is the code for our papers:
 - [Deep Variational Instance Segmentation](https://arxiv.org/abs/2007.11576)

The implementation is based on YOLACT's implementation at [Yolact-github](https://github.com/dbolya/yolact)


# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone  xxx.git
   cd yolact
   ```
 - Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     - Run `conda env create -f environment.yml`
   - Manually with pip
     - Set up a Python3 environment (e.g., using virtenv).
     - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
     - Install some other packages:
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install cython
       pip install opencv-python pillow pycocotools matplotlib 
       pip install scikit-image
       pip install scipy==1.2.0
       ```
 - If you'd like to train DVIS, download the COCO dataset and the 2014/2017 annotations. Note that this script will take a while and dump 21gb of files into `./data/coco`.
   ```Shell
   sh data/scripts/COCO.sh
   ```
 - If you'd like to evaluate DVIS on `test-dev`, download `test-dev` with this script.
   ```Shell
   sh data/scripts/COCO_test.sh
   ```
 - If you want to use YOLACT++ base, compile deformable convolutional layers (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0)).
   Make sure you have the latest CUDA toolkit installed from [NVidia's Website](https://developer.nvidia.com/cuda-toolkit).
   ```Shell
   cd external/DCNv2
   python setup.py build develop
   ```

## Train on your own dataset:
 - You could edit the config_xx.py in data/ to customize the network setting and dataset setting.
 - You could run with specific the arguments on shell command:
   ```Shell
    python train.py --config=plus_resnet50_config_550 --resume=PATH/TO/YOUR/FILE --start_iter=0 --exp_name=dummy     
   ```
 - Or, you could customize the json script in exp_scripts/, and run with:
   ```Shell
    python train.py --scripts=exp_scripts/xxx.json
   ```

