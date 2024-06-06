## YoloV10 export onnx

1. 下载源码

   ```bash
   git clone https://github.com/THU-MIG/yolov10
   cd yolov10
   pip install -e .
   ```

2. 修改 ultralytics/engine/exporter.py 如下

   ```python
   # line 369
   output_names = ["output0", "output1"] if isinstance(self.model, SegmentationModel) else ["output"]
   dynamic = True
           if dynamic:
               dynamic = {'images': {0: 'batch'}} 
               if isinstance(self.model, SegmentationModel):
                   dynamic['output0'] = {0: 'batch', 2: 'anchors'} 
                   dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'} 
               elif isinstance(self.model, DetectionModel):
                   dynamic['output'] = {0: 'batch'} 
   ```

3. 导出onnx

   ```bash
   yolo export model=yolov10s.pt format=onnx
   ```



