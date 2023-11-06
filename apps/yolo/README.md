## Export onnx models

### YoloV5 export onnx

1. 下载源码

   ```bash
   https://github.com/ultralytics/yolov5.git
   git chechout v6.0
   ```

2. 修改部分forward代码

   ```python
   # line 55 forward function in yolov5/models/yolo.py 
   # bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
   # modified into:
   bs, _, ny, nx = map(int, x[i].shape)  # x(bs,255,20,20) to x(bs,3,20,20,85)
   bs = -1
   
   # line 65 in yolov5/models/yolo.py
   # if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
   #    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
   # modified into:
   if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
       self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
   anchor_grid = (self.anchors[i].clone() * self.stride[i]).view(1, -1, 1, 1, 2) # disconnect for pytorch trace
   
   # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
   # modified into:
   y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
   
   # wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
   # modified into:
   wh = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
   
   #  z.append(y.view(bs, -1, self.no))
   # modified into：
   z.append(y.view(bs, self.na * ny * nx, self.no))
   
   # line 52 in yolov5/export.py
   # torch.onnx.export(dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
   #                                'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)  修改为
   # modified into:
   torch.onnx.export(dynamic_axes={'images': {0: 'batch'}, 
                                   'output': {0: 'batch'} }
   ```

3. 导出onnx

   ```bash
   cd yolov5
   python export.py --weights=yolov5s.pt --dynamic --include=onnx --opset=13
   ```

### YoloX export onnx

1. 下载源码

   ```bash
   https://github.com/Megvii-BaseDetection/YOLOX.git
   git chechout 0.1.0
   ```

2. 修改部分forward代码

   ```python
   # line 206 forward fuction in yolox/models/yolo_head.py. Replace the commented code with the uncommented code
   # self.hw = [x.shape[-2:] for x in outputs] 
   self.hw = [list(map(int, x.shape[-2:])) for x in outputs]
   
   
   # line 208 forward function in yolox/models/yolo_head.py. Replace the commented code with the uncommented code
   # [batch, n_anchors_all, 85]
   # outputs = torch.cat(
   #     [x.flatten(start_dim=2) for x in outputs], dim=2
   # ).permute(0, 2, 1)
   proc_view = lambda x: x.view(-1, int(x.size(1)), int(x.size(2) * x.size(3)))
   outputs = torch.cat(
       [proc_view(x) for x in outputs], dim=2
   ).permute(0, 2, 1)
   
   
   # line 253 decode_output function in yolox/models/yolo_head.py Replace the commented code with the uncommented code
   #outputs[..., :2] = (outputs[..., :2] + grids) * strides
   #outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
   #return outputs
   xy = (outputs[..., :2] + grids) * strides
   wh = torch.exp(outputs[..., 2:4]) * strides
   return torch.cat((xy, wh, outputs[..., 4:]), dim=-1)
   
   # line 77 in tools/export_onnx.py
   model.head.decode_in_inference = True
   ```

3. 导出onnx

   ```bash
   cd YOLOX
   export PYTHONPATH=$PYTHONPATH:.
   python tools/export_onnx.py -c yolox_s.pth -f exps/default/yolox_s.py --output-name=yolox_s.onnx --dynamic --no-onnxsim
   ```

### YoloV7 export onnx

1. 下载源码

   ```bash
   https://github.com/WongKinYiu/yolov7.git
   git chechout v0.1
   ```

2. 修改部分forward代码（类似yolov5）

   ```python
   # line 45 forward function in yolov7/models/yolo.py 
   # bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
   # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
   # modified into:
   
   bs, _, ny, nx = map(int, x[i].shape)  # x(bs,255,20,20) to x(bs,3,20,20,85)
   bs = -1
   x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
   
   # line 52 in yolov7/models/yolo.py
   # y = x[i].sigmoid()
   # y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
   # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
   # z.append(y.view(bs, -1, self.no))
   # modified into：
   y = x[i].sigmoid()
   xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
   wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, -1, 1, 1, 2)  # wh
   classif = y[..., 4:]
   y = torch.cat([xy, wh, classif], dim=-1)
   z.append(y.view(bs, self.na * ny * nx, self.no))
   
   # line 57 in yolov7/models/yolo.py
   # return x if self.training else (torch.cat(z, 1), x)
   # modified into:
   return x if self.training else torch.cat(z, 1)
   
   
   # line 52 in yolov7/models/export.py
   # output_names=['classes', 'boxes'] if y is None else ['output'],
   # dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
   #               'output': {0: 'batch', 2: 'y', 3: 'x'}} if opt.dynamic else None)
   # modified into:
   output_names=['classes', 'boxes'] if y is None else ['output'],
   dynamic_axes={'images': {0: 'batch'},  # size(1,3,640,640)
                 'output': {0: 'batch'}} if opt.dynamic else None)
   ```

3. 导出onnx

   ```bash
   cd yolov7
   python models/export.py --dynamic --grid --weight=yolov7.pt
   ```

### YoloV8 export onnx

1. 下载源码

   ```bash
   https://github.com/ultralytics/ultralytics.git
   cd ultralytics
   python setup.py develop
   ```

2. 新建 export.py 文件如下

   ```python
   from ultralytics import YOLO
   # 加载模型
   model = YOLO("yolov8s.pt")  # 加载预训练模型（建议用于训练）
   success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
   ```

3. 修改 ultralytics/engine/exporter.py 如下

   ```python
   # line 313
   # dynamic = self.args.dynamic
   dynamic = True
           if dynamic:
               dynamic = {'images': {0: 'batch'}}  # shape(1,3,640,640)
               if isinstance(self.model, SegmentationModel):
                   dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 116, 8400)
                   dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
               elif isinstance(self.model, DetectionModel):
                   dynamic['output'] = {0: 'batch', 2: 'anchors'}  # shape(1, 84, 8400)
   ```

4. 导出onnx

   ```bash
   python export.py
   ```

5. 将输出转置

   ```python
   # 新建v8trans.py
   import onnx
   import onnx.helper as helper
   import sys
   import os
   
   def main():
       if len(sys.argv) < 2:
           print("Usage:\n python v8trans.py yolov8n.onnx")
           return 1
       file = sys.argv[1]
       if not os.path.exists(file):
           print(f"Not exist path: {file}")
           return 1
       prefix, suffix = os.path.splitext(file)
       dst = prefix + ".transd" + suffix
       model = onnx.load(file)
       node = model.graph.node[-1]
       old_output = node.output[0]
       node.output[0] = "pre_transpose"
       for specout in model.graph.output:
           if specout.name == old_output:
               shape0 = specout.type.tensor_type.shape.dim[0]
               shape1 = specout.type.tensor_type.shape.dim[1]
               shape2 = specout.type.tensor_type.shape.dim[2]
               new_out = helper.make_tensor_value_info(
                   specout.name,
                   specout.type.tensor_type.elem_type,
                   [0, 0, 0]
               )
               new_out.type.tensor_type.shape.dim[0].CopyFrom(shape0)
               new_out.type.tensor_type.shape.dim[2].CopyFrom(shape1)
               new_out.type.tensor_type.shape.dim[1].CopyFrom(shape2)
               specout.CopyFrom(new_out)
   
       model.graph.node.append(
           helper.make_node("Transpose", ["pre_transpose"], [old_output], perm=[0, 2, 1])
       )
       print(f"Model save to {dst}")
       onnx.save(model, dst)
       return 0
   
   if __name__ == "__main__":
       sys.exit(main())
   ```

   ```bash
   # 执行
   python v8trans.py yolov8s.onnx
   ```



