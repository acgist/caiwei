# CUDA

* 版本: 12.6
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

## 模型转换
## 相关链接

* https://developer.nvidia.com/cuda-downloads
* https://developer.nvidia.com/cuda-toolkit-archive
* https://developer.nvidia.com/cuda/cuda-x-libraries/cudnn
* https://developer.download.nvidia.cn/compute/cudnn/redist/cudnn
