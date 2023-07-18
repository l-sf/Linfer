echo ""
echo ""
echo "************ compile yolov5 models ***************"
echo ""
trtexec --onnx=./onnx_models/yolov5s.onnx \
		--saveEngine=./yolov5s.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16

echo ""
trtexec --onnx=./onnx_models/yolov5s_ptq.onnx \
		--saveEngine=./yolov5s_ptq.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--int8

echo ""
trtexec --onnx=./onnx_models/yolov5s_qat.onnx \
		--saveEngine=./yolov5s_qat.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--int8

trtexec --onnx=./onnx_models/yolov5m.onnx \
		--saveEngine=./yolov5m.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16

echo ""
trtexec --onnx=./onnx_models/yolov5m_ptq.onnx \
		--saveEngine=./yolov5m_ptq.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--int8


echo ""
echo ""
echo "************ compile yolox models ***************"
echo ""
trtexec --onnx=./onnx_models/yolox_s.onnx \
		--saveEngine=./yolox_s.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16

echo ""
trtexec --onnx=./onnx_models/yolox_m.onnx \
		--saveEngine=./yolox_m.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16


echo ""
echo ""
echo "************ compile yolov7 models ***************"
echo ""
trtexec --onnx=./onnx_models/yolov7.onnx \
		--saveEngine=./yolov7.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16

echo ""
trtexec --onnx=./onnx_models/yolov7_qat.onnx \
		--saveEngine=./yolov7_qat.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--int8


echo ""
echo ""
echo "************ compile yolov8 models ***************"
echo ""
trtexec --onnx=./onnx_models/yolov8n.onnx \
		--saveEngine=./yolov8n.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16

echo ""
trtexec --onnx=./onnx_models/yolov8s.onnx \
		--saveEngine=./yolov8s.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16

echo ""
trtexec --onnx=./onnx_models/yolov8m.onnx \
		--saveEngine=./yolov8m.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16
