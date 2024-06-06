

# Linfer

![Language](https://img.shields.io/badge/language-c++-brightgreen) ![Language](https://img.shields.io/badge/CUDA-12.1-brightgreen) ![Language](https://img.shields.io/badge/TensorRT-8.6.1.6-brightgreen) ![Language](https://img.shields.io/badge/OpenCV-4.5.5-brightgreen) ![Language](https://img.shields.io/badge/ubuntu-20.04-brightorigin)

## Introduction

基于 TensorRT 的 C++ 高性能推理库。



## Update News

🚀（2024.06.06）支持目标检测算法Yolov10！

🚀（2024.05.23）支持语义分割算法：PaddleSeg中的PP-LiteSeg、MobileSeg，轻量高效，适合部署！

🚀（2023.12.03）支持全景驾驶感知算法 YOLOPv2，Better、Faster、Stronger ！

🚀（2023.11.06）支持全景驾驶感知算法 YOLOP ！

🚀（2023.10.19）支持单目标跟踪 OSTrack、LightTrack ！单独的单目标跟踪仓库为 [github](https://github.com/l-sf/Track-trt) 

🚀（2023.10.09）支持目标检测算法 RT-DETR ！

🚀（2023.08.26）支持 PTQ 量化，Yolov5/7 QAT 量化！

🚀（2023.07.19）支持目标检测 Yolo 系列 5/X/7/8，多目标跟踪 Bytetrack。



## Highlights

- 支持全景驾驶感知 YOLOPv2，目标检测 RT-DETR，Yolov5/X/7/8/10 ，多目标跟踪 Bytetrack，单目标跟踪 OSTrack、LightTrack；
- 预处理和后处理实现CUDA核函数，在 jetson 边缘端也能高性能推理；
- 封装Tensor、Infer，实现内存复用、CPU/GPU 内存自动拷贝、引擎上下文管理、输入输出绑定等；
- 推理过程实现生产者消费者模型，实现预处理和推理的并行化，进一步提升性能；
- 采用 RAII 思想+接口模式封装应用，使用安全、便捷。



## Easy Using

本项目代码结构如下：`apps` 文件夹中存放着各个算法的实现代码，其中 `app_xxx.cpp` 是对应 `xxx` 算法的调用demo函数，每个算法彼此之间没有依赖，假如只需要使用yolopv2，可以将此文件夹下的其他算法全部删除，没有影响；`trt_common` 文件夹中包括了常用的cuda_tools，对TensorRT进行Tensor、Infer的封装，生产者消费者模型的封装；`quant-tools` 文件夹中是量化脚本，主要是yolov5/7；`workspace` 文件夹中存放编译好的可执行文件、engine等。

使用哪个算法就在 `main.cpp` 中调用哪个算法的demo函数。

```bash
.
├── apps
│   ├── yolo
│   └── yolop
│   ├── app_yolo.cpp
│   ├── app_yolop.cpp
│   ├── ...
├── trt_common
│   ├── cuda_tools.hpp
│   ├── trt_infer.hpp
│   ├── trt_tensor.hpp
│   └── ...
├── quant-tools
│   └── ...
├── workspace
│   └── ...
├── CMakeLists.txt
└── main.cpp
```

如果要进行您自己的算法部署，只需要在 `apps` 文件夹中新建您的算法文件夹，模仿其他算法中对 `trt_infer/trt_tensor` 等的使用即可。后续时间空闲较多的情况下会更新较为详细的用法。



## Project Build and Run

1. install cuda/tensorrt/opencv

   [reference](https://github.com/l-sf/Notes/blob/main/notes/Ubuntu20.04_install_tutorials.md#%E4%BA%94cuda--cudnn--tensorrt-install) 

2. compile engine

   1. 下载 onnx 模型 [google driver](https://drive.google.com/drive/folders/16ZqDaxlWm1aDXQsjsxLS7yFL0YqzHbxT?usp=sharing) 或者按照教程导出，教程在各文件夹READEME

   2. ```bash
      cd Linfer/workspace
      # 修改其中的onnx路径
      bash compile_engine.sh
      ```

3. build 

   ```bash
   # 修改 CMakeLists.txt 中 cuda/tensorrt/opencv 为自己的路径
   cd Linfer
   mkdir build && cd build
   cmake .. && make -j4
   ```

4. run

   ```bash
   cd Linfer/workspace
   ./pro
   ```



## Speed Test

在 Jetson Orin Nano 8G 上进行测试，测试包括整个流程（图像预处理+模型推理+后处理解码）

|   Model    | Precision | Resolution | FPS(bs=1) |
| :--------: | :-------: | :--------: | :-------: |
|  yolov5_s  |   fp16    |  640x640   |   96.06   |
|  yolox_s   |   fp16    |  640x640   |   79.64   |
|   yolov7   | **int8**  |  640x640   |   49.55   |
|  yolov8_n  |   fp16    |  640x640   |  121.94   |
|  yolov8_s  |   fp16    |  640x640   |   81.40   |
|  yolov8_l  |   fp16    |  640x640   |    13     |
| yolov10_n  |   fp16    |  640x640   |           |
| yolov10_s  |   fp16    |  640x640   |           |
| yolov10_l  |   fp16    |  640x640   |           |
| rtdetr_r50 |   fp16    |  640x640   |    12     |
| lighttrack |   fp16    |  256x256   |   90.91   |
|  ostrack   |   fp16    |  256x256   |   37.04   |
|   yolop    |   fp16    |  640x640   |   31.4    |
|  yolopv2   |   fp16    |  480x640   |   21.9    |



## Reference

- [tensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro.git) 
- [Video：详解TensorRT的C++/Python高性能部署，实战应用到项目](https://www.bilibili.com/video/BV1Xw411f7FW/?share_source=copy_web&vd_source=4bb05d1ac6ff39b7680900de14419dca) 

