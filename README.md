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

Please download [20 US Cities dataset](https://github.com/songtaohe/Sat2Graph/tree/master/prepare_dataset) and organize them as following:

```
code_root/
└── data/
    └── 20cities/
```

After downloading the dataset run the following script to preprocess and prepare the data for training
```
python generate_data.py
```

## 2. Training

#### 2.1 Prepare config file

The config file can be found at `.configs/road_rgb_2D.yaml`. Make custom changes if necessary.

#### 2.2.a Training on multiple-GPU (e.g. 3 GPUs)

For example, the command for training Relationformer is following:

```bash
python train.py --config configs/road_rgb_2D.yaml --cuda_visible_device 0 1 2 --nproc_per_node 3
```

<!-- #### 2.2.b Training on slurm cluster (e.g. 3 GPUs)

If you are using slurm cluster, you can simply run the following command to train on 1 node:

```bash
srun -u --nodelist worker-1 --gres=gpu:3 -c 16 python train.py --config configs/synth_3D.yaml --nproc_per_node=3
``` -->

## 3. Evaluation

Once you have the config file and trained model of Relation, run following command to evaluate it on test set:

```bash
python test.py --config configs/road_rgb_2D.yaml --checkpoint ./trained_weights/last_checkpoint.pt
```

## 4. Interactive notebook

Please find the `debug_relationformer.ipynb` for interactive evaluation and visualization
