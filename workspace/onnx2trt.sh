#trtexec --onnx=../../yolov5-6.2/weights/yolov5s.onnx \
#		--saveEngine=./yolov5s.trt \
#		--buildOnly \
#		--minShapes=images:1x3x640x640 \
#		--optShapes=images:1x3x640x640 \
#		--maxShapes=images:8x3x640x640 \
#		--fp16


#trtexec --onnx=../../yolov5-6.2/weights/ptq/yolov5s_ptq.onnx \
#		--saveEngine=./yolov5s_ptq.trt \
#		--buildOnly \
#		--minShapes=images:1x3x640x640 \
#		--optShapes=images:1x3x640x640 \
#		--maxShapes=images:8x3x640x640 \
#		--int8


trtexec --onnx=../../yolov5-6.2/weights/qat/yolov5s_qat.onnx \
		--saveEngine=./yolov5s_qat.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--int8