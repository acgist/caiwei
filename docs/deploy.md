# 部署

[依赖安装](../deps/README.md)

```
git clone https://github.com/acgist/caiwei.git
cd caiwei
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
sudo make install
```

## CUDA

* https://developer.nvidia.com/cuda-downloads

```
nvidia-smi
```

## Conda

* https://www.anaconda.com/download

```
conda create -n caiwei python=3.12
```

## Torch

* https://pytorch.org/get-started/locally/

```
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126
```

## ModelScope

* https://www.modelscope.cn/docs/home

```
pip install accelerate modelscope transformers --index-url https://download.pytorch.org/whl/cu126
```
