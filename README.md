# Fast Dynamic Graph Attention with Initial Residual and Identity Mapping

This repository contains a PyTorch implementation of "FDGATII : Fast Dynamic Graph Attention with Initial Residual and Identity Mapping".(https://arxiv.org/abs/2110.11464)

The repo, inlcuding data and datasplits used for the 10 ierations, has been forked initially from [GCNII](https://github.com/chennnM/GCNII). We use the sparse (static) GATv1 attention code from [pyGAT](https://github.com/Diego999/pyGAT) and modified for dynamic attention as in [GATv2](https://arxiv.org/abs/2105.14491). 

## Dependencies
- CUDA 11.3.0
- python 3.6.9
- pytorch 1.3.1
Note : FDGATII is able to run with no GPU if the GUP timing code is commented out, and then will not require CUDA. 

## Datasets

The `data` folder contains three benchmark datasets(Cora, Citeseer, Pubmed), and the `newdata` folder contains four datasets(Chameleon, Cornell, Texas, Wisconsin) from [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn). We use the same same full-supervised setting, and data splits,  as Geom-GCN and GCNII. 

## Results
Testing accuracy summarized below. We have used 10 data splits ( Split 0 - 9) and obtained the Avarage Accuracy and Standard deviation.
| Dataset | Depth |Dimensions|  Accuracy | Std.D |Split 0      | Split 1      | Split 2      | Split 3      | Split 4      | Split 5      | Split 6      | Split 7      | Split 8      | Split 9 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Cora | 2 |64  | 87.7867 | 1.149  | 87.1227 | 89.1348 | 88.7324 | 87.7264 | 87.7264 | 86.3179 | 85.5131 | 89.1348 | 87.7264 | 88.7324 |
| Cite | 1 |128 | 75.6434 | 1.8721 | 73.7237 | 75.0751 | 75.5255 | 74.9249 | 79.2453 | 73.5849 | 74.1742 | 78.979  | 75.6757 | 75.5255 |
| Pubm | 2 |64  | 90.3524 | 0.297  | 90.5426 | 90.644  | 90.213  | 89.858  | 90.5426 | 90.8215 | 90.4412 | 90.2383 | 89.8834 | 90.3398 |
| Cham | 1 |64  | 65.1754 | 1.8105 | 67.1053 | 66.2281 | 61.4035 | 63.5965 | 65.5702 | 64.4737 | 64.2544 | 64.693  | 67.9825 | 66.4474 |
| Corn | 1 |128 | 82.4324 | 6.3095 | 67.5676 | 83.7838 | 91.8919 | 86.4865 | 86.4865 | 83.7838 | 83.7838 | 83.7838 | 75.6757 | 81.0811 |
| Texa | 2 |64  | 80.5405 | 5.2407 | 75.6757 | 81.0811 | 81.0811 | 91.8919 | 75.6757 | 81.0811 | 78.3784 | 72.973  | 81.0811 | 86.4865 |
| Wisc | 1 |128 | 86.2745 | 4.4713 | 86.2745 | 80.3922 | 88.2353 | 92.1569 | 90.1961 | 82.3529 | 82.3529 | 84.3137 | 82.3529 | 94.1176 |


## Usage
- All parameters are defined in fullSupervised_01.py.

- To run FDGATII on cora, for 1 iteration only use
```
python -u fullSupervised_01.py --data cora --layer 2 --alpha 0.2 --weight_decay 1e-4 --epochs 1500 --iterations 1 --mode FDGATII --support 1 --verbosity 1 --model GCNII_BASE
```

- To replicate the FDGATII full-supervised results, run the following script
```sh
#!/bin/bash
SCRIPT='python3.6 fullSupervised_01.py'
for SUPPORT in 1 2
do

SETTTINGS=" --epochs 1500 --iterations 10 --mode FDGATII --support $SUPPORT --verbosity 0 --model GCNII_BASE "

$SCRIPT --data cora --layer 2 --hidden 64 --alpha 0.2 --weight_decay 1e-4 $SETTTINGS
$SCRIPT --data citeseer --layer 1 --hidden 128  --weight_decay 5e-6 $SETTTINGS
$SCRIPT --data pubmed --layer 2 --hidden 64  --alpha 0.1 --weight_decay 5e-6 $SETTTINGS
$SCRIPT --data chameleon --layer 1 --hidden 64 --lamda 1.5 --alpha 0.2 --weight_decay 5e-4 $SETTTINGS
$SCRIPT --data cornell --layer 1 --hidden  128 --lamda 1 --weight_decay 1e-3 $SETTTINGS
$SCRIPT --data texas --layer 2 --hidden 64 --lamda 1.5 --weight_decay 1e-4 $SETTTINGS
$SCRIPT --data wisconsin --layer 1 --hidden 128 --lamda 1 --weight_decay 5e-4 $SETTTINGS

done
```

## Citation
```
@article{kulatilleke2021fdgatii,
  title={FDGATII: Fast Dynamic Graph Attention with Initial Residual and Identity Mapping},
  author={Kulatilleke, Gayan K and Portmann, Marius and Ko, Ryan and Chandra, Shekhar S},
  journal={arXiv preprint arXiv:2110.11464},
  year={2021}
}
```
