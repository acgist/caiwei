# WSL

## 安装WSL

```
# 安装
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
# 重启
shutdown /r /t 0
# 卸载
dism.exe /online /disable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /disable-feature /featurename:VirtualMachinePlatform /all /norestart
# 重启
shutdown /r /t 0
```

## 安装Linux

```
# 常见命令
wsl --status
wsl --version
# 设置默认版本
wsl --set-default-version 2
# 更新运行时和内核
wsl --update [--web-download]
# 查看已安装的版本
wsl -l -v
# 查看可安装的版本
wsl -l -o
# 在线安装
wsl --install -d Ubuntu-24.04 [--web-download]
# 离线安装
# https://github.com/microsoft/WSL/blob/master/distributions/DistributionInfo.json
wsl --install --from-file D:\download\ubuntu-24.04.5-wsl-amd64.wsl --location D:\develop\wsl\Ubuntu-24.04
# 进入
wsl
wsl -d Ubuntu-24.04
# 关闭实例
wsl --terminate Ubuntu-24.04
# 卸载实例
wsl --unregister Ubuntu-24.04
# 关闭全部
wsl --shutdown
```
