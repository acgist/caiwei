# 依赖项目

## SDL2

* 版本：2.32.10

```
# APT安装
apt install libsdl2-dev

# VCPKG安装
vcpkg install sdl2
vcpkg export  sdl2 --zip
```

## FFmpeg

* 版本：6.1.1

```
# APT安装
apt install ffmpeg

# VCPKG安装
vcpkg install ffmpeg
vcpkg export  ffmpeg --zip

# 编译安装
apt install nasm yasm libx264-dev

git clone https://github.com/FFmpeg/nv-codec-headers.git
cd nv-codec-headers
git switch sdk/12.1
make -j4
sudo make install

wget http://www.ffmpeg.org/releases/ffmpeg-6.1.1.tar.xz
tar -Jxvf ffmpeg-6.1.1.tar.xz
cd ffmpeg-6.1.1/
PKG_CONFIG_PATH="/usr/local/lib/pkgconfig/"
./configure          \
  --enable-gpl       \
  --enable-static    \
  --enable-shared    \
  --enable-cuda      \
  --enable-cuvid     \
  --enable-nvenc     \
  --enable-libnpp    \
  --enable-libx264   \
  --enable-nonfree   \
  --enable-cuda-nvcc \
  --extra-cflags="-I/usr/local/cuda/include" --extra-ldflags="-L/usr/local/cuda/lib64"
make -j4
sudo make install
```

## cpp-httplib

* 版本：0.47.0

```
# APT安装
apt install libcpp-httplib-dev

# VCPKG安装
vcpkg install cpp-httplib
vcpkg export  cpp-httplib --zip

# 编译安装
git clone -b v0.47.0 --depth=1 https://github.com/yhirose/cpp-httplib.git
cd cpp-httplib
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
sudo make install
```
