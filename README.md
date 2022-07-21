# Relationformer: A Unified Framework for Image-to-Graph Generation

## Requirements
* CUDA>=9.2
* PyTorch>=1.7.1

For other system requirements please follow

```bash
pip install -r requirements.txt
```

### Compiling CUDA operators
```bash
cd ./models/ops
python setup.py install
```


## Code Usage

## 1. Dataset preparation

Please download [Toulouse Road Network dataset](https://github.com/davide-belli/toulouse-road-network-dataset) by following the steps under **Usage**. The structure of the dataset should be as follows:

```
code_root/
└── data/
    toulouse-road-network/
    └── augment/
        └── images
    └── augment.pickle
    └── test/
        └── images
    └── test_images.pickle
    └── test.pickle
    └── train/
        └── images
    └── train_images.pickle
    └── train.pickle
    └── valid/
        └── images
    └── valid_images.pickle
    └── valid.pickle
```

## 2. Training

#### 2.1 Prepare config file

The config file can be found at `.configs/road_2D.yaml`. Make custom changes if necessary.

#### 2.2 Train

For example, the command for training Relationformer is following:

```bash
python train.py --config configs/road_2D.yaml --cuda_visible_device 3
```

## 3. Evaluation

Once you have the config file and trained model, run following command to evaluate it on test set:

```bash
python test.py --config configs/road_2D.yaml --cuda_visible_device 3 --checkpoint ./trained_weights/last_checkpoint.pt
```