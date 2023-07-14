trtexec --onnx=../../yolov5-6.0/weights/yolov5s.onnx \
		--saveEngine=./yolov5s.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:4x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16