**Relationformer**: A Unified Framework for Image-to-Graph Generation
========

[![DOI](https://img.shields.io/badge/arXiv-https%3A%2F%2Fdoi.org%2F10.48550%2FarXiv.2203.10202-B31B1B)](https://doi.org/10.48550/arXiv.2203.10202) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![image-to-graph](.github/problem_statment.png "Logo Title Text 1")

**What is image-to-graph?**  Image-to-graph is a general class of problem appearing in many forms in computer vision and medical imaging. Primarily, the task is to discover an image's underlying structural- or semantic- graph representation. A few examples are shown above.

In spatio-structural tasks, such as road network extraction (Fig. a), nodes represent road-junctions
or significant turns, while edges correspond to structural connections, i.e., the road itself. Similarly, in 3D blood vessel-graph extraction (Fig. b), nodes represent branching points or substantial curves, and edges correspond to structural connections, i.e., arteries, veins, and capillaries.

In the case of spatio-semantic graph generation, e.g., scene graph generation from natural images (Fig. c), the objects denote nodes, and the semantic relation denotes the edges.

 Note that the 2D road network extraction and 3D vessel graph extraction tasks have undirected relations while the scene graph generation task has directed relations.


![Relationformer](.github/comparison.png )

**What it is Relationformer?** Relationformer is a unified one-stage transformer-based framework that jointly predicts objects and their relations. We leverage direct set-based object prediction and incorporate the interaction among the objects to learn an object-relation representation simultaneously. In addition to existing [obj]-tokens, we propose a novel learnable token, namely the [rln]-token. Together with [obj]-tokens, [rln]-token exploits local and global semantic reasoning in an image through a series of mutual associations. In combination with the pair-wise [obj]-token, the [rln]-token contributes to a computationally efficient relation prediction.

A pictorial depiction above illustrates a general architectural evolution of transformers in computer vision and how Relationformer advances the concept of a task-specific learnable token one step further. The proposed Relationformer is also shown compared to the conventional two-stage relation predictor. The amalgamation of two separate stages not only simplifies the architectural pipeline but also co-reinforces both of the tasks.

**About the code**. We have provided the source code of Relation former along with instructions for training and evaluation script for individual datasets. Please follow the procedure below.


## Usage

### Repository Structure

Please navigate through the respective branch for specific application

```
relationformer
├── master           #overview
|   └──
|   └──
├── vessel_graph     #3D vessel graph dataset
|   └──
|   └──
└── road_network     #2D binary road network dataset
|   └──
|   └──
└── road_network_rgb #2D RGB road network dataset
|   └──
|   └──
└── scene_graph      #2D scene graph dataset
    └──
    └──
```

We have instructions for individual dataset in their respective branch.

## Citing Relationformer
If you find our repository useful in your research, please cite the following:
```bibtex
@article{shit2022relationformer,
  title={Relationformer: A Unified Framework for Image-to-Graph Generation},
  author={Shit, Suprosanna and Koner, Rajat and Wittmann, Bastian and others},
  journal={arXiv preprint arXiv:2203.10202},
  year={2022}
}
```

# License
Relationformer code is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

# Acknowledgement

We acknowledge the following repository from where we have inherited code snippets

1. DETR: [[code](https://github.com/facebookresearch/detr)][[paper](https://arxiv.org/abs/2005.12872)]
2. Deformable-DETR: [[code](https://github.com/fundamentalvision/Deformable-DETR)][[paper](https://arxiv.org/abs/2010.04159)]
3. RTN: [[code](https://github.com/rajatkoner08/rtn)][[paper](https://arxiv.org/abs/2004.06193)]
4. GGT: [[code](https://github.com/davide-belli/generative-graph-transformer)][[paper](https://arxiv.org/abs/1910.14388)]

<!-- # Contributing
We actively welcome your pull requests! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for more info. -->
