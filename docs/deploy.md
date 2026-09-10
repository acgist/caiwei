# 部署

* [基础依赖](../deps/README.md)
* [模型部署](model)
* [推理引擎](runtime)

```
git clone https://github.com/acgist/caiwei.git
cd caiwei
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
sudo make install
```

* `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`
* `-DCMAKE_C_COMPILER=/usr/bin/gcc-14 -DCMAKE_CXX_COMPILER=/usr/bin/g++-14`

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
* https://modelscope.cn/organization/Qwen
* https://modelscope.cn/organization/ggml-org
* https://modelscope.cn/collections/acgist/caiwei

```
pip install ms_swift accelerate modelscope transformers --index-url https://download.pytorch.org/whl/cu126
```

> `ms_swift==4.4.2`

## Linux环境

```
# 版本管理
sudo apt install cmake build-essential

sudo apt install gcc-11 g++-11
sudo apt install gcc-12 g++-12
sudo apt install gcc-14 g++-14

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 14
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 14

sudo update-alternatives --list       gcc
sudo update-alternatives --config     gcc
sudo update-alternatives --display    gcc
sudo update-alternatives --remove-all gcc

# 编译安装
# wget http://ftp.gnu.org/gnu/gcc/gcc-14.2.0/gcc-14.2.0.tar.gz
wget https://mirrors.aliyun.com/gnu/gcc/gcc-14.2.0/gcc-14.2.0.tar.gz
tar -zxvf gcc-14.2.0.tar.gz
cd gcc-14.2.0
# ./contrib/download_prerequisites
sudo apt install libgmp-dev libmpc-dev libmpfr-dev

mkdir build
cd    build
../configure -v --prefix=/usr/local/gcc-14.2.0 --disable-multilib --enable-checking=release --enable-languages=c,c++
make -j4
sudo make install

sudo update-alternatives --install /usr/bin/gcc gcc /usr/local/gcc-14.2.0/bin/gcc-14.2.0 14
sudo update-alternatives --install /usr/bin/g++ g++ /usr/local/gcc-14.2.0/bin/g++-14.2.0 14
```

* `export LD_LIBRARY_PATH="/usr/local/gcc-14.2.0/lib64/:$LD_LIBRARY_PATH"`
* `-DCMAKE_C_COMPILER=/usr/local/gcc-14.2.0/bin/gcc-14.2.0 -DCMAKE_CXX_COMPILER=/usr/local/gcc-14.2.0/bin/g++-14.2.0`

## Windows环境

* https://cmake.org/download/
* https://vcpkg.io/en/index.html
* https://code.visualstudio.com/
* https://visualstudio.microsoft.com/zh-hans/vs/
