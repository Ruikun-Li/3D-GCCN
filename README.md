# 3D-GCCN
This is the reference code for ["3D Graph-Connectivity Constrained Network for Hepatic Vessel Segmentation"](https://ieeexplore.ieee.org/abstract/document/9562259)

## Dependency
* Python 3.6+
* Pytorch 1.5+
* NetworkX 2.4
* scikit-fmm 2019.1.30

## Vascular connectivity graph construction
1. Prepare the data and convert the images and labels to the same size through preprocessing.
2. Set the *sampling interval* and *travel time threshold* according to the specific task (such as image size, etc.)
3. Run `1-graph_onstruction.py` to construct 3D vascular connectivity graphs for each image.

## Training
1. Set network and training parameters according to specific task. As our method is *backbone-agnostic*, the encoder and decoder of CNN part can be replaced by any more powerful segmentation network.
2. Run `2-train.py` to train the 3D-GCNN. The segmentation network (CNN) is trained under the supervision of connectivity by GNN.

## Testing
1. Run `3-test.py` to test the segmentation network. Here, the GNN is not used benefitting from the *plug-in mode*, which greatly reduces hardware and time costs in the inference stage and is more suitable for 3D images and clinical practice.

## Citation
```
@ARTICLE{9562259,
  author={Li, Ruikun and Huang, Yi-Jie and Chen, Huai and Liu, Xiaoqing and Yu, Yizhou and Qian, Dahong and Wang, Lisheng},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={3D Graph-Connectivity Constrained Network for Hepatic Vessel Segmentation}, 
  year={2022},
  volume={26},
  number={3},
  pages={1251-1262},
  doi={10.1109/JBHI.2021.3118104}
}
```
