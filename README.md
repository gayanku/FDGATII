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
Testing accuracy summarized below.
| Dataset | Depth |Dimensions|  Accuracy | 
|:---:|:---:|:---:|:---:|
| Cora | 2 |64  | 87.7867 | 
| Cite | 1 |128 | 75.6434 | 
| Pubm | 2 |64  | 90.3524 |
| Cham | 1 |64  | 65.1754 |
| Corn | 1 |128 | 82.4324 |
| Texa | 2 |64  | 80.5405 |
| Wisc | 1 |128 | 86.2745 |



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
