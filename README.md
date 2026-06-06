# MLGO

This is a PyTorch implementation of the paper: [MLGO: Multi-Layer graph neural ODEs for traffic forecasting - PubMed](https://pubmed.ncbi.nlm.nih.gov/41558194/)published in Neural Networks-2026.

## Requirements

The model is implemented using Python3 with dependencies as followed:
tqdm==4.67.3

torch==2.11.0

torchdiffeq==0.2.5

fastdtw==0.3.4

numpy==2.4.4

## Data Preparation

### PeMS Datasets

Download data folder ,which contains PeMS03, PeMS04, PeMS07, PeMS08, and PeMS-BAY datasets from [https://github.com/guoshnBJTU/ASTGNN](https://github.com/guoshnBJTU/ASTGNN). Uncompress them.

## Model Training

PeMS03    

```
python run.py --config ./configs/pems03.conf
```

PeMS04    

```
python run.py --config ./configs/pems04.conf
```

PeMS07    

```
python run.py --config ./configs/pems07.conf
```

PeMS08    

```
python run.py --config ./configs/pems08.conf
```

PeMS-BAY    

```
python run.py --config ./configs/pemsbay.conf
```

## Citation

```
@article{gao2026connecting,
  title={MLGO: Multi-Layer graph neural ODEs for traffic forecasting},
  author={Gao, Mengzhou and Yu, Huangqian and Jiao, Pengfei},
  journal={Neural Networks},
  year={2026}
}
```
