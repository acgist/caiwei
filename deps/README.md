# 依赖项目

## SDL2

**编译测试时才需要安装**

* 版本: 2.32.10
* 官网: https://www.libsdl.org/

```
# APT安装
sudo apt install libsdl2-dev

# VCPKG安装
vcpkg install sdl2:x64-windows
vcpkg export  sdl2:x64-windows --zip
```

## FFmpeg

* 版本: 6.1.1
* 官网: https://ffmpeg.org/

```
# APT安装
sudo apt install ffmpeg libavutil-dev libavcodec-dev libavdevice-dev libavformat-dev libswscale-dev libswresample-dev

# VCPKG安装
vcpkg install ffmpeg:x64-windows
vcpkg export  ffmpeg:x64-windows --zip

# 编译安装
sudo apt install nasm yasm libx264-dev
wget http://www.ffmpeg.org/releases/ffmpeg-6.1.1.tar.xz
tar -Jxvf ffmpeg-6.1.1.tar.xz
cd ffmpeg-6.1.1
PKG_CONFIG_PATH="/usr/local/lib/pkgconfig/"
./configure        \
  --enable-gpl     \
  --enable-shared  \
  --enable-libx264 \
  --enable-nonfree
make -j4
sudo make install
# 配置RKMPP支持
--enable-rkmpp --enable-libdrm --enable-version3
```

## cpp-httplib

* 版本: 0.58.0
* 官网: https://github.com/yhirose/cpp-httplib

```
# APT安装
sudo apt install libcpp-httplib-dev

# VCPKG安装
vcpkg install cpp-httplib:x64-windows
vcpkg export  cpp-httplib:x64-windows --zip

# 编译安装
git clone -b v0.58.0 --depth=1 https://github.com/yhirose/cpp-httplib.git
cd cpp-httplib
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
sudo make install
```
