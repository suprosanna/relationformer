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

Please download [Visual Genome dataset](https://visualgenome.org/), [Stanford correction](https://github.com/rowanz/neural-motifs/tree/master/data/stanford_filtered), [other files](https://drive.google.com/drive/folders/1ClYMg-1EHbf7ap4N-7aBoJlXkRtlKVa_?usp=sharing) (VG.pt, obj_count.pkl) for reproducibility and organize them as following:

```
code_root/
└── data/
    └── visual_genome/
        ├── VG.pt
        ├── obj_count.pkl
        ├── VG_100K/
        |    ├── 1.jpg
        |    ├── 2.jpg
        |    ├── ...
        |    └── ...
        └── stanford_filtered/
             ├── image_data.json
             ├── proposals.h5
             ├── VG-SGG-dicts.json
             └── VG-SGG.h5

```

## 2. Training

#### 2.1 Prepare config file

The config file can be found at `.configs/scene_2d.yaml`. Make custom changes if necessary.

### 2.2.a Instruction for multi-gpu training e.g. using GPU 2
`python3 train.py --config configs/scene_2d.yaml --cuda_visible_device 0 1 2 --nproc_per_node 3 --b 16 `

### 2.2.b Instruction for training with SLURM
`srun -u --nodelist worker-2 --gres=gpu:1 -c 16 python3 train.py --config configs/scene_2d.yaml --nproc_per_node 1 --b 16`

### 2.2.c Distributed training e.g. using 4 GPUs
`srun -u --nodelist worker-1 --gres=gpu:4 -c 16 python3 train.py --config configs/scene_2d.yaml --nproc_per_node=4`

## 3. Evaluation

Once you have the config file and trained model of Relation, run following command to evaluate it on test set:

```bash
python run_batch_inference_eval.py --config configs/scene_2d --model ./trained_weights/last_checkpoint.pt --eval
```

## 4. Interactive notebook

Please find the `debug_relationformer.ipynb` for interactive evaluation and visualization

## Extra  
The result on scene graph is reported with ***ResNet-50*** as backbone and tested on Visual Genome dataset. However we incorporated [SWIN ](https://arxiv.org/pdf/2103.14030.pdf) transformer backbone and [Open Image](https://storage.googleapis.com/openimages/web/index.html) dataset for large scale experiment.
