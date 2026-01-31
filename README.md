<p style="" align="center">
  <img src="./assets/logo.png" alt="Logo" width="20%">
</p>
<h1 align="center">PyNvVideoPipe</h1>
<p style="margin:0px" align="center">
    <img src="https://img.shields.io/badge/license-BSD--2-blue.svg?&logo=c&logoColor=white&style=for-the-badge">
    <img src="https://img.shields.io/badge/CUDA-12.8-76B900?&logo=nvidia&logoColor=white&style=for-the-badge">
    <img src="https://img.shields.io/badge/OS-Linux-FCC624?&logo=linux&logoColor=white&style=for-the-badge">
</p>

---

基于 NVIDIA CUDA 的 Python 高性能视频处理流水线实现

⭐ 多进程绕过 GIL 限制，支持多流、多 GPU 与多模型推理

⭐ 减少 Host-Device 数据拷贝和 GPU 显存冗余拷贝，提升推理效率

⭐ 开箱即用，扩展性强，适合中小型项目快速部署

|                                                           | Open Source 开源 |     Learning Curve 学习成本      | Developer Friendliness 二次开发友好度 | Performance 性能 |
| :-------------------------------------------------------: | :--------------: | :------------------------------: | :-----------------------------------: | :--------------: |
| [DeepStream](https://developer.nvidia.com/deepstream-sdk) |        ❌         |               High               |                  Low                  |       High       |
| [VideoPipe](https://github.com/sherlockchou86/VideoPipe)  |        ✅         | medium（requires cpp knowledge） |   Medium（requires cpp knowledge）    |      Medium      |
|                            Our                            |        ✅         |               ≈ 0                |           High +++++++++++            |      Medium      |

## Quick Start

本项目推荐 Docker 容器运行，首先确保本地环境满足以下三个条件：

- Docker >= 24.0.0
- NVIDIA Driver >= 590
- NVIDIA Container Toolkit >= 1.13.0

### 1. 生成镜像

clone 本项目，生成包含完整开发环境的镜像

```bash
git clone https://github.com/lmk123568/PyNvVideoPipe.git
cd PyNvVideoPipe/docker
docker build -t PyNvVideoPipe:cuda12.8 .
```

镜像生成后，进入容器，不报错即成功

```bash
docker run -it \
  --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v {your_path}/PyNvVideoPipe:/workspace \
  PyNvVideoPipe:cuda12.6 \
  bash
```

后续示例代码默认在容器内`/workspace`运行

> ⚠️ 不推荐自己本地装环境，如果一定要自己装，请参考 Dockerfile

### 2. 编译硬件编解码库

```bash
python scripts/setup.py install
```

### 3. 训练模型权重转换

将通过 [ultralytics](https://github.com/ultralytics/ultralytics) 训练的`pt`模型导入到当前目录下（示例模型为 [yolo26n.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt)）

```bash
python scripts/pt2trt.py  --w ./yolo26n.pt --fp16
```

> 💡 推理尺寸建议`(576,1024)`，可以跳过`letterbox`降低计算开销

### 4. 运行

开启 MPS（Multi-Process Service）

```bash
nvidia-cuda-mps-control -d
# echo quit | nvidia-cuda-mps-control  关闭 MPS
```

阅读理解其代码并运行

```bash
python main.py
```

## Benchmark

测试日期: 2026-01-25

测试硬件: AMD Ryzen 9 5950 X + NVIDIA GeForce RTX 3090

测试任务: 4 × RTSP Decoders → YOLO26 (TensorRT) → 4 × RTMP Encoders

|                           | CPU     | RAM     | GPU VRAM | **GPU-Util** |
| ------------------------- | ------- | ------- | -------- | ------------ |
| VidepPipe（ffmpeg codec） | 511.6 % | 1.5 GiB | 2677 MiB | 16 %         |
| Our                       | 40 %    | 1.2GiB  | 3932 MiB | 12 %         |

> 工程不是追求完美的数学解，而是在资源受限、时间紧迫、需求模糊的情况下，寻找一个可用的最优解

## Pipeline

<img src="./assets/pipeline.png" alt="pipe" style="zoom:70%;" />

## License

[BSD-2-Clause](https://github.com/lmk123568/PyNvVideoPipe/blob/main/LICENSE)
