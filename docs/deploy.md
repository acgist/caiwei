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

-DENABLE_CAIWEI_RUNTIME_ONNXRUNTIME=OFF -DENABLE_CAIWEI_BACKEND_RKNN=ON -DENABLE_CAIWEI_RUNTIME_RKNN2=ON -DENABLE_CAIWEI_RUNTIME_RKNN3=ON
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
pip install ms_swift==4.4.2 accelerate modelscope transformers --index-url https://download.pytorch.org/whl/cu126
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

# 编译安装
# wget http://ftp.gnu.org/gnu/gcc/gcc-15.3.0/gcc-15.3.0.tar.gz
wget https://mirrors.aliyun.com/gnu/gcc/gcc-15.3.0/gcc-15.3.0.tar.gz
tar -zxvf gcc-15.3.0.tar.gz
cd gcc-15.3.0
# ./contrib/download_prerequisites
sudo apt install libgmp-dev libmpc-dev libmpfr-dev

mkdir build
cd build
../configure -v --prefix=/usr/local/gcc-15.3.0 --disable-multilib --disable-bootstrap --enable-checking=release --enable-languages=c,c++
make -j4
sudo make install

sudo update-alternatives --install /usr/bin/gcc gcc /usr/local/gcc-15.3.0/bin/gcc 15
sudo update-alternatives --install /usr/bin/g++ g++ /usr/local/gcc-15.3.0/bin/g++ 15
```

* `export LD_LIBRARY_PATH="/usr/local/gcc-15.3.0/lib64/:$LD_LIBRARY_PATH"`
* `-DCMAKE_C_COMPILER=/usr/local/gcc-15.3.0/bin/gcc -DCMAKE_CXX_COMPILER=/usr/local/gcc-15.3.0/bin/g++`
* 经过多次测试发现`ARM`架构`GCC-14.2.0`/`GCC-14.4.0`这些版本使用`generator`存在`BUG`容易导致`RKNN3`推理失败。

## Windows环境

* https://cmake.org/download/
* https://vcpkg.io/en/index.html
* https://code.visualstudio.com/
* https://visualstudio.microsoft.com/zh-hans/vs/
