# CUDA

* 版本: 12.6
* 版本: 13.4
* 官网: https://www.nvidia.cn

## 安装教程

```
# Linux
sudo apt install cuda‑12‑6
sudo apt install cuda-toolkit-12-6

# Windows
1. 安装驱动`Studio Driver`
2. 安装工具`CUDA Toolkit`
3. 安装工具`NVIDIA cuDNN`

# 验证
nvcc -V
nvidia-smi
```

> `CMake`配置`nvcc`路径`-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`

|硬件版本|架构代号|显卡型号|
|:--|:--|:--|
|120|Blackwell   |`RTX PRO 6000/4500` `RTX 5090/5080/5070 Ti/5070/5060 Ti/5060/5050`|
|110|Blackwell   |`Jetson T5000` `Jetson T4000`|
|103|Blackwell   |`GB300` `B300`|
|100|Blackwell   |`GB200` `B200`|
|90 |Hopper      |`GH200` `H200` `H100`|
|89 |Ada Lovelace|`L4/L40/L40S` `RTX 6000/5000/4500/4000/2000 Ada` `RTX 4090/4080/4070 Ti/4070/4060 Ti/4060/4050`|
|87 |Ampere      |`Jetson AGX Orin` `Jetson Orin NX` `Jetson Orin Nano`|
|86 |Ampere      |`RTX 3090/3080 Ti/3080/3070 Ti/3070/3060 Ti/3060/3050`|
|80 |Ampere      |`A100` `A30` `A10`|
|75 |Turing      |`RTX 2080 Ti/2080/2070/2060` `T4` `Quadro RTX 8000/6000/5000/4000`|
|70 |Volta       |`Tesla V100` `Titan V`|
|61 |Pascal      |`GTX 1080 Ti/1080/1070 Ti/1070/1060/1050 Ti/1050` `P40` `P4`|
|60 |Pascal      |`Tesla P100`|
|52 |Maxwell     |`GTX 980 Ti/980/970/960` `GTX Titan X`|
|50 |Maxwell     |`GTX 750 Ti/750`|
|35 |Kepler      |`GTX Titan` `Tesla K80/K40/K20`|
|30 |Kepler      |`GTX 680/670/660/650`|

## 模型转换
## 相关链接

* https://developer.nvidia.com/cuda-downloads
* https://developer.nvidia.com/cuda-toolkit-archive
* https://developer.nvidia.com/cuda/cuda-x-libraries/cudnn
* https://developer.download.nvidia.cn/compute/cudnn/redist/cudnn
