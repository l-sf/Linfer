

# Linfer

![Language](https://img.shields.io/badge/language-c++-brightgreen) ![Language](https://img.shields.io/badge/CUDA-12.1-brightgreen) ![Language](https://img.shields.io/badge/TensorRT-8.6.1.6-brightgreen) ![Language](https://img.shields.io/badge/OpenCV-4.5.5-brightgreen) ![Language](https://img.shields.io/badge/ubuntu-20.04-brightorigin)

## Introduction

基于 TensorRT 的 C++ 高性能推理库。



## Update News

🚀（2023.11.06）支持全景驾驶感知算法 YOLOP ！

🚀（2023.10.19）支持单目标跟踪 OSTrack、LightTrack ！单独的单目标跟踪仓库为 [github](https://github.com/l-sf/Track-trt) 

🚀（2023.10.09）支持目标检测算法 RT-DETR ！

🚀（2023.08.26）支持 PTQ 量化，Yolov5/7 QAT 量化！

🚀（2023.07.19）支持目标检测 Yolo 系列 5/X/7/8，多目标跟踪 Bytetrack。



## Highlights

- 支持全景驾驶感知 YOLOP，目标检测 RT-DETR，Yolo 5/X/7/8 ，多目标跟踪 Bytetrack，单目标跟踪 OSTrack、LightTrack；
- 预处理和后处理实现CUDA核函数，在 jetson 边缘端也能高性能推理；
- 封装Tensor、Infer，实现内存复用、CPU/GPU 内存自动拷贝、引擎上下文管理、输入输出绑定等；
- 推理过程实现生产者消费者模型，实现预处理和推理的并行化，进一步提升性能；
- 采用 RAII 思想+接口模式封装应用，使用安全、便捷。



## Easy Using

**3 lines of code to implement yolo inference**

```c++
auto infer = Yolo::create_infer("yolov5s.trt", Yolo::Type::V5, 0); 
auto image = cv::imread("imgs/bus.jpg");
auto boxes = infer->commit(image).get();
```



## Project Build and Run

1. install cuda/tensorrt/opencv

   [reference](https://github.com/l-sf/Notes/blob/main/notes/Ubuntu20.04_install_tutorials.md#%E4%BA%94cuda--cudnn--tensorrt-install) 

2. compile engine

   1. 下载onnx模型 [google driver](https://drive.google.com/drive/folders/16ZqDaxlWm1aDXQsjsxLS7yFL0YqzHbxT?usp=sharing) 或 按照下面的教程自己导出

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

在 Jetson Orin Nano 8G 上进行测试，测试包括整个流程（即预处理+推理+后处理）

|   Model    | Precision | Resolution | FPS(bs=1) | FPS(bs=4) |
| :--------: | :-------: | :--------: | :-------: | :-------: |
|  yolov5_s  |   fp16    |  640x640   |   96.06   |   100.9   |
|  yolox_s   |   fp16    |  640x640   |   79.64   |   85.00   |
|   yolov7   |   int8    |  640x640   |   49.55   |   50.42   |
|  yolov8_n  |   fp16    |  640x640   |  121.94   |  130.16   |
|  yolov8_s  |   fp16    |  640x640   |   81.40   |   84.74   |
|  yolov8_l  |   fp16    |  640x640   |    13     |     -     |
| rtdetr_r50 |   fp16    |  640x640   |    12     |     -     |
| lighttrack |   fp16    |  256x256   |   90.91   |     -     |
|  ostrack   |   fp16    |  256x256   |   37.04   |     -     |
|   yolop    |   fp16    |  640x640   |   31.4    |     -     |



## Reference

- [tensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro.git) 
- [infer](https://github.com/shouxieai/infer.git) 
- [Video：详解TensorRT的C++/Python高性能部署，实战应用到项目](https://www.bilibili.com/video/BV1Xw411f7FW/?share_source=copy_web&vd_source=4bb05d1ac6ff39b7680900de14419dca) 

