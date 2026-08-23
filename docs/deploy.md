# 部署

```
cmake .. -DCMAKE_BUILD_TYPE=Release
make
make -j4
make install
```

## CUDA

```
nvidia-smi
```

## Conda

```
conda create -n caiwei python=3.12
```

## Torch

```
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126
```

## ModelScope

```
pip install accelerate modelscope transformers --index-url https://download.pytorch.org/whl/cu126

微调工具
```
