# 部署

* [基础依赖](../deps/README.md)
* [推理依赖](runtime)

```
git clone https://github.com/acgist/caiwei.git
cd caiwei
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
sudo make install
```

* 指定`CUDA`编译器: `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`
* 指定`GCC/G++`编译器: `-DCMAKE_C_COMPILER=/usr/bin/gcc-14 -DCMAKE_CXX_COMPILER=/usr/bin/g++-14`

## CUDA

* https://developer.nvidia.com/cuda-downloads
* https://developer.nvidia.com/cuda-toolkit-archive

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
pip install torch torchaudio torchcodec torchvision --index-url https://download.pytorch.org/whl/cu126
```

## ModelScope

* https://www.modelscope.cn/docs/home

```
pip install accelerate modelscope transformers --index-url https://download.pytorch.org/whl/cu126
```

## 日志前缀

```
* = 修改代码
+ = 增加功能
- = 减少功能
~ = 优化代码
% = 重要更新
@ = 修复问题
! = 版本发布
$ = 配置更新
# = 文档更新
& = 依赖升级
? = 其他修改
```

## Linux环境

```
# 版本管理
sudo apt install cmake build-essential

sudo apt install gcc-11 g++-11
sudo apt install gcc-12 g++-12

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12

sudo update-alternatives --list       gcc
sudo update-alternatives --config     gcc
sudo update-alternatives --display    gcc
sudo update-alternatives --remove-all gcc

sudo update-alternatives --list       cuda
sudo update-alternatives --config     cuda
sudo update-alternatives --display    cuda
sudo update-alternatives --remove-all cuda

# 安装新版
# wget http://ftp.gnu.org/gnu/gcc/gcc-14.2.0/gcc-14.2.0.tar.gz
wget https://mirrors.aliyun.com/gnu/gcc/gcc-14.2.0/gcc-14.2.0.tar.gz
tar -zxvf gcc-14.2.0.tar.gz
cd gcc-14.2.0
./contrib/download_prerequisites

mkdir build
cd    build
../configure -v --prefix=/usr/local/gcc-14.2.0 --disable-multilib --enable-checking=release --enable-languages=c,c++
make -j4
sudo make install

sudo update-alternatives --install /usr/bin/gcc gcc /usr/local/gcc-14.2.0/bin/gcc-14.2.0 14
sudo update-alternatives --install /usr/bin/g++ g++ /usr/local/gcc-14.2.0/bin/g++-14.2.0 14
```

> `-DCMAKE_C_COMPILER=/usr/local/gcc-14.2.0/bin/gcc-14.2.0 -DCMAKE_CXX_COMPILER=/usr/local/gcc-14.2.0/bin/g++-14.2.0`

## Windows环境

* https://cmake.org/download/
* https://vcpkg.io/en/index.html
* https://code.visualstudio.com/
* https://visualstudio.microsoft.com/zh-hans/vs/
