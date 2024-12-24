# Linfer 

![Language](https://img.shields.io/badge/language-c++-brightgreen) ![Language](https://img.shields.io/badge/CUDA-12.1-brightgreen) ![Language](https://img.shields.io/badge/TensorRT-8.6.1.6-brightgreen) ![Language](https://img.shields.io/badge/OpenCV-4.5.5-brightgreen) ![Language](https://img.shields.io/badge/ubuntu-20.04-brightorigin)

English | [简体中文](README.md)

## Introduction 
A high-performance inference library for C++ based on TensorRT.



## Update News
🚀(2024.12.24) Supports YAML Configuration

🚀(2024.06.06) Supports target detection algorithm Yolov10!

🚀(2024.05.23) Supports semantic segmentation algorithm: PP-LiteSeg and MobileSeg in PaddleSeg, which are lightweight and efficient and suitable for deployment!

🚀(2023.12.03) Supports Panoramic driving perception algorithm YOLOPv2, Better, Faster, Stronger!

🚀 (2023.11.06) Support panoramic driving perception algorithm YOLOP!

🚀 (2023.10.19) Support single target tracking OSTrack, LightTrack! The separate single target tracking repository is [github]( https://github.com/l-sf/Track-trt)

🚀(2023.10.09) Support target detection algorithm RT-DETR!

🚀(2023.08.26) Support PTQ quantization, Yolov5/7 QAT quantization!

🚀(2023.07.19) Support target detection Yolo series 5/X/7/8, multi-target tracking Bytetrack.

## Highlights

- Support panoramic driving perception YOLOPv2, Target detection RT-DETR, Yolov5/X/7/8/10, multi-target tracking Bytetrack, single target tracking OSTrack, LightTrack;
- Pre-processing and post-processing implement CUDA kernel functions, and high-performance reasoning can also be achieved on the Jetson edge;
- Encapsulate Tensor and Infer to achieve memory reuse, automatic CPU/GPU memory copying, engine context management, input and output binding, etc.;
- The inference process implements the producer-consumer model, realizes the parallelization of preprocessing and inference, and further improves performance; - Use RAII concept + interface mode to encapsulate applications, which is safe and convenient to use.

## Easy Using

The code structure of this project is as follows: The implementation code of each algorithm is stored in the `apps` folder, where `app_xxx.cpp` is the call demo function corresponding to the `xxx` algorithm. Each algorithm has no dependency on each other. If you only need to use yolopv2, you can delete all other algorithms in this folder without any impact; the `trt_common` folder includes the commonly used cuda_tools, which encapsulates TensorRT's Tensor and Infer, and the producer-consumer model; The `quant-tools` folder contains quantitative scripts, mainly yolov5/7.

Which algorithm to use is called in `main.cpp` demo function.

```bash
.
├── apps
│   ├── yolo
│   └── yolop
│   ├── app_yolo.cpp
│   ├── app_yolop.cpp
│   ├── ...
├── trt_common
│   ├── cuda_tools. hpp
│   ├── trt_infer.hpp
│   ├── trt_tensor.hpp
│   └── ...
├── quant-tools
│   └── ...
├── workspace
│ └── ...
├── CMakeLists .txt
└── main.cpp
```

If you want to deploy your own algorithm, just create a new folder for your algorithm in the `apps` folder, and imitate the `trt_infer/trt_tensor` in other algorithms. You can use it as you like. I will update more detailed instructions later when I have more free time.


## Project Build and Run

1. install cuda/tensorrt/opencv

   [reference](https://github.com/l-sf/Notes/blob/main/notes/Ubuntu20.04_install_tutorials.md#%E4%BA%94cuda--cudnn--tensorrt-install) 

2. compile engine

3. Download the onnx model from [google drive](https://drive.google.com/drive/folders/16ZqDaxlWm1aDXQsjsxLS7yFL0YqzHbxT?usp=sharing) or export it according to the tutorial, the tutorial is in README of each folder. Put your onnx file under `workspace/onnx_models` folder (create it)
   
 ```bash
    cd Linfer/workspace
    # Modify the onnx path bash compile_engine.sh

    # Uncomment particular model form compile_engine.sh or copy any of the commands from it like this

    # YOLOV8S
    trtexec --onnx=./onnx_models/yolov8n.onnx \
		--saveEngine=./yolov8n.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16
```

4. build

```bash
# Modify CMakeLists.txt cuda/tensorrt/opencv is your own path cd Linfer
mkdir build && cd build
cmake .. && make -j4
```

5. Configure: make your configuration file config.yaml please see a demo of file in config.yaml for instance if you wnt to run bytetrack tracking algorithm with YOLOV8, you'll need to build .trt from step 3 and then provide it's path in config.yaml like below

```yaml
tasks:
  - task: "track"
    subtasks:
      - type: "inference_bytetrack"
        engine_file: "/home/e300/mahmood/code/Linfer/workspace/yolov8s.trt"
        gpuid: 0
        yolo_type: "V8"
        video_file: "/home/e300/mahmood/code/Linfer/workspace/videos/snow.mp4"
        output_save_path: ""
```

6. run (to avoid any errors please provide full paths always)

```bash
cd Linfer/workspace
./pro config.yaml
```

## Speed Test

Tested on Jetson Orin Nano 8G, the test includes the entire process (image preprocessing + model inference + post-processing decoding)

|   Model    | Precision | Resolution | FPS(bs=1) |
| :--------: | :-------: | :--------: | :-------: |
|  yolov5_s  |   fp16    |  640x640   |   96.06   |
|  yolox_s   |   fp16    |  640x640   |   79.64   |
|   yolov7   | **int8**  |  640x640   |   49.55   |
|  yolov8_n  |   fp16    |  640x640   |  121.94   |
|  yolov8_s  |   fp16    |  640x640   |   81.40   |
|  yolov8_m  |   fp16    |  640x640   |   41.14   |
|  yolov8_l  |   fp16    |  640x640   |   27.52   |
| yolov10_n  |   fp16    |  640x640   |  115.13   |
| yolov10_s  |   fp16    |  640x640   |   73.65   |
| yolov10_m  |   fp16    |  640x640   |   39.51   |
| yolov10_l  |   fp16    |  640x640   |   26.41   |
| rtdetr_r50 |   fp16    |  640x640   |   11.25   |
| lighttrack |   fp16    |  256x256   |   90.91   |
|  ostrack   |   fp16    |  256x256   |   37.04   |
|   yolop    |   fp16    |  640x640   |   31.4    |
|  yolopv2   |   fp16    |  480x640   |   21.9    |
| PP-LiteSeg |   fp16    |  256x512   |  129.81   |
| MobileSeg  |   fp16    |  256x512   |  140.36   |



## Reference

- [tensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro.git) 
- [Video：详解TensorRT的C++/Python高性能部署，实战应用到项目](https://www.bilibili.com/video/BV1Xw411f7FW/?share_source=copy_web&vd_source=4bb05d1ac6ff39b7680900de14419dca) 

