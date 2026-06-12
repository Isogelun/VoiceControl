# Unitree Go2 Deploy

本文是 Go2 场景的部署速查。完整说明见 [robot-production-deploy.md](robot-production-deploy.md)。

## 1. 打包

Windows 项目根目录：

```powershell
scripts\build_robot_deploy_bundle.bat
```

不带模型：

```powershell
scripts\build_robot_deploy_bundle.bat -NoModels
```

如果机器狗本地 ONNX Runtime 不兼容，生成外部推理配置：

```powershell
scripts\build_robot_deploy_bundle.bat -Backend external
```

## 2. 拷贝

```bash
scp -r dist/robot-deploy-<timestamp> unitree@10.10.20.82:~/
```

## 3. 安装

```bash
ssh unitree@10.10.20.82
cd ~/robot-deploy-<timestamp>
chmod +x scripts/install_robot_target.sh
scripts/install_robot_target.sh --mode webrtc
```

## 4. 启动

```bash
./start_robot.sh
```

## 5. external 模式

编辑 `config.deploy.yaml`：

```yaml
inference:
  backend: external

services:
  asr_url: http://<上位机IP>:8000/asr
  nlu_url: http://<上位机IP>:8001/nlu
```

然后：

```bash
./start_robot.sh
```

## 6. 动作服务

默认动作服务地址通常是：

```yaml
command:
  service_url: http://10.10.20.82:8090/api/v1/local/motion
```

如果机器狗 IP 变了，需要同步改这里。
