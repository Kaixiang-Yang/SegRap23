# Solution of Team kxyang for SegRap2023 Challenge
Our methods for two sub-Tasks (Task1: OARs; Task2: GTVs) are built upon [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).

## Environments and Weights
We provide docker version here, if you want to run in server, please set as follow. For more details like path setting, please refer to [nnU-Netv1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).

```bash
download nnU-NetV1
cd nnUNet
pip install -e .
```
The weight will be released latter, put them in .../SegRap2023_Task1/weight, the dir tree is like:

    SegRap2023_Task1
    ├── weight
    │   └── all
    │       ├── model_final_checkpoint.model
    │       ├── model_final_checkpoint.model.pkl
    │       └── plans.pkl
    ...
    
## 1.Task001 OARs segmentation

### docker
```bash
sh .../SegRap2023_Task1/build.sh
sh .../SegRap2023_Task1/export.sh
```

### server
```bash
cd .../SegRap2023_Task1
python process_local.py
```

## 2.Task002 GTVs segmentation

### docker
```bash
sh .../SegRap2023_Task2/build.sh
sh .../SegRap2023_Task2/export.sh
```

### server
```bash
cd .../SegRap2023_Task2
python process_local.py
```
