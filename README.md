# PyNvVideoPipe: 全流程 GPU 视频处理框架 (Full-Process GPU Video Pipeline)

PyNvVideoPipe 是一个专为高性能计算机视觉任务设计的 Python 框架，旨在通过**全流程 GPU 驻留**彻底消除 CPU-GPU 数据传输瓶颈。

传统的视频分析流水线（Decode -> CPU -> GPU Inference -> CPU -> Encode）往往因为频繁的内存拷贝而受限于带宽和延迟。PyNvVideoPipe 通过深度集成 NVIDIA 硬件编解码能力与 PyTorch/TensorRT 生态，实现了从视频摄取到最终推流的**零拷贝 (Zero-Copy)** 处理。

## ✨ 核心特性 (Key Features)

*   **🚀 全流程 GPU 操作 (End-to-End GPU Pipeline)**
    视频帧从解码开始，经过预处理、推理、追踪、绘图，直到最终编码输出，数据始终驻留在显存 (VRAM) 中。

*   **🎥 硬件解码 (Hardware Decoding)**
    基于 FFmpeg 与 NVIDIA NVDEC 的 C++ 绑定，直接将 RTSP/本地视频流解码为 GPU Tensors，无缝对接 PyTorch。

*   **⚡ TensorRT 推理 (TensorRT Inference)**
    内置高性能 TensorRT 后端，支持 YOLO 等主流模型（FP16/INT8），充分释放 Tensor Core 算力，实现毫秒级检测。

*   **🎯 目标跟踪 (Object Tracking)**
    支持接入 GPU 优化的跟踪算法，在显存中直接处理检测结果，维持超高帧率的实时跟踪。

*   **🎨 GPU 绘图 (On-GPU Drawing)**
    利用 CUDA/NPP 在 GPU 上直接绘制检测框 (Bounding Boxes) 和元数据，避免了回传 CPU 使用 OpenCV 绘图的巨大性能开销。

*   **💾 硬件编码 (Hardware Encoding)**
    处理完成的画面直接调用 NVENC 进行硬件编码并推流，极大降低 CPU 占用。

## 🛠️ 快速开始 (Quick Start)

(这里可以补充安装和运行代码示例)
