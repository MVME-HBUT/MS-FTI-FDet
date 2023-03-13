# MS FTI-FDet

MS FTI-FDet is a lightweight and high high accuracy detector for freight trains fault detection.

Source code for  **Visual Fault Detection of Multi-Scale Key Components in Freight Trains**. For more details, please refer to our [paper](https://ieeexplore.ieee.org/document/996418).

The source code is based on [CornerNet](https://github.com/princeton-vl/CornerNet) and [MatrixNet](https://github.com/arashwan/matrixnet).

## Getting Started

### Installing Packages

#### Using Conda

Please first install Anaconda and create an Anaconda environment using the provided package list.

```
conda create --name <name> --file packagelist_conda.txt
```

After one creates the environment, activate it.

```
source activate<name>
```

#### Using Pip

Alternatively, one can use pip and install all packages from the requirements file. Note we are using python 3.6+. Torch 1.2.0 and torchvision 0.4.0

```
pip install -r requirements.txt
```

Our current implementation only supports GPU, so one needs a GPU and need to have CUDA(9+)  installed on your machine.

### Compiling NMS

You also need to compile the NMS code (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/cpu_nms.pyx) and [Soft-NMS](https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx)).

```
cd <dir>/external
make
```

### Using Data

Use self-built COCO type datasets and store them in `<dir>/data/coco/images/`

## Training and Evaluation

To train and evaluate a network, one will need to create a configuration file, which defines the hyperparameters, and a model file, which defines the network architecture. The configuration file should be in JSON format and placed in `config/`. Each configuration file should have a corresponding model file in `models/` (specified by `model_name` in the config file). i.e.

To train a model:

```
python train.py <config_file>
```

```
python test.py ACH_MatrixNetAnchorsResnet50 --testiter 100000 --split testing --debug
```

`--debug` flag can be used to save the first 200 images with detections under results directory.

## Cite

```
@ARTICLE{9964185,
  author={Zhang, Yang and Zhou, Yang and Pan, Huilin and Wu, Bo and Sun, Guodong},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Visual Fault Detection of Multi-Scale Key Components in Freight Trains}, 
  year={2022},
  volume={},
  number={},
  pages={1-9},
  doi={10.1109/TII.2022.3224989}}

```
