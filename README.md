## 1. get onnx
export onnx:
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
# 640
python export.py --weights=./weights/yolov5n.pt --include=onnx  
```

## 2.compile onnx
```bash
cd yolov5
trtexec --onnx=./weights/yolov5n.onnx --saveEngine=./weights/yolov5n.trt --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --fp16
```
## 3.run
```bash
mkdir build
cd build
cmake ..
make -j8
# infer an image
./app_yolov5  --version=v560 --model=../yolov5/weights/yolov5n.trt   --size=640  --batch_size=1 --img=../data/dog.jpg  --savePath 
# infer video
./app_yolov5  --version=v560 --model=../yolov5/weights/yolov5n.trt   --size=640  --batch_size=8 --video=../data/people.mp4 --show --savePath=../
# infer web camera
./app_yolov5  --version=v560 --model=../yolov5/weights/yolov5n.trt   --size=640  --batch_size=4 --cam_id=0 --show --savePath
```









## 5. appendix

offical weights for yolov5.6.0<br>
[yolov5s]   |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt<br>
[yolov5m]   |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt<br>
[yolov5l]   |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l.pt<br>
[yolov5x]   |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5x.pt<br>
[yolov5s6]  |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s6.pt<br>
[yolov5m6]  |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m6.pt<br>
[yolov5l6]  |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l6.pt<br>
[yolov5x6]  |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5x6.pt<br>