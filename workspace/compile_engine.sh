# echo ""
# echo ""
# echo "************************ compile yolov5 models ***************************"
# echo ""
# trtexec --onnx=./onnx_models/yolov5s.onnx \
# 		--saveEngine=./yolov5s.trt \
# 		--buildOnly \
# 		--minShapes=images:1x3x640x640 \
# 		--optShapes=images:1x3x640x640 \
# 		--maxShapes=images:8x3x640x640 \
# 		--fp16

# echo ""
# trtexec --onnx=./onnx_models/yolov5m.onnx \
# 		--saveEngine=./yolov5m.trt \
# 		--buildOnly \
# 		--minShapes=images:1x3x640x640 \
# 		--optShapes=images:1x3x640x640 \
# 		--maxShapes=images:8x3x640x640 \
# 		--fp16


# echo ""
# echo ""
# echo "************************ compile yolox models ***************************"
# echo ""
# trtexec --onnx=./onnx_models/yolox_s.onnx \
# 		--saveEngine=./yolox_s.trt \
# 		--buildOnly \
# 		--minShapes=images:1x3x640x640 \
# 		--optShapes=images:1x3x640x640 \
# 		--maxShapes=images:8x3x640x640 \
# 		--fp16

# echo ""
# trtexec --onnx=./onnx_models/yolox_m.onnx \
# 		--saveEngine=./yolox_m.trt \
# 		--buildOnly \
# 		--minShapes=images:1x3x640x640 \
# 		--optShapes=images:1x3x640x640 \
# 		--maxShapes=images:8x3x640x640 \
# 		--fp16


# echo ""
# echo ""
# echo "************************ compile yolov7 models ***************************"
# echo ""
# trtexec --onnx=./onnx_models/yolov7.onnx \
# 		--saveEngine=./yolov7.trt \
# 		--buildOnly \
# 		--minShapes=images:1x3x640x640 \
# 		--optShapes=images:1x3x640x640 \
# 		--maxShapes=images:8x3x640x640 \
# 		--fp16

# echo ""
# trtexec --onnx=./onnx_models/yolov7_qat.onnx \
# 		--saveEngine=./yolov7_qat.trt \
# 		--buildOnly \
# 		--minShapes=images:1x3x640x640 \
# 		--optShapes=images:1x3x640x640 \
# 		--maxShapes=images:8x3x640x640 \
# 		--int8


echo ""
echo ""
echo "************************ compile yolov8 models ***************************"
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

echo ""
trtexec --onnx=./onnx_models/yolov8l.onnx \
		--saveEngine=./yolov8l.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16


echo ""
echo ""
echo "************************ compile yolov10 models ***************************"
echo ""
trtexec --onnx=./onnx_models/yolov10n.onnx \
		--saveEngine=./yolov10n.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16

echo ""
trtexec --onnx=./onnx_models/yolov10s.onnx \
		--saveEngine=./yolov10s.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16

echo ""
trtexec --onnx=./onnx_models/yolov10m.onnx \
		--saveEngine=./yolov10m.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16

echo ""
trtexec --onnx=./onnx_models/yolov10l.onnx \
		--saveEngine=./yolov10l.trt \
		--buildOnly \
		--minShapes=images:1x3x640x640 \
		--optShapes=images:1x3x640x640 \
		--maxShapes=images:8x3x640x640 \
		--fp16


# echo ""
# echo ""
# echo "************************ compile RT-DETR models ***************************"
# echo ""
# trtexec --onnx=./onnx_models/rtdetr_r50vd_6x_coco_dynamic.onnx \
# 		--saveEngine=./rtdetr_r50vd_6x_coco_dynamic_fp16.trt \
# 		--buildOnly \
#     --minShapes=image:1x3x640x640 \
#     --optShapes=image:1x3x640x640 \
#     --maxShapes=image:8x3x640x640 \
#     --fp16


# echo ""
# echo ""
# echo "************************ compile LightTrack models ***************************"
# echo ""
# trtexec --onnx=./onnx_models/lighttrack-z.onnx \
# 		--saveEngine=./lighttrack-z.trt \
# 		--buildOnly \
# 		--fp16

# echo ""
# trtexec --onnx=./onnx_models/lighttrack-x-head.onnx \
# 		--saveEngine=./lighttrack-x-head.trt \
# 		--buildOnly \
# 		--fp16


# echo ""
# echo ""
# echo "************************ compile OSTrack models ***************************"
# echo ""
# trtexec --onnx=./onnx_models/ostrack-256.onnx \
# 		--saveEngine=./ostrack-256.trt \
# 		--buildOnly \
# 		--fp16

# echo ""
# trtexec --onnx=./onnx_models/ostrack-384-ce.onnx \
# 		--saveEngine=./ostrack-384-ce.trt \
# 		--buildOnly \
# 		--fp16


# echo ""
# echo ""
# echo "************************ compile YoloP model ***************************"
# echo ""
# trtexec --onnx=./onnx_models/yolop-640.onnx \
# 		--saveEngine=./yolop-640.trt \
# 		--buildOnly \
# 		--fp16

# echo ""
# trtexec --onnx=./onnx_models/yolopv2-480x640.onnx \
# 		--saveEngine=./yolopv2-480x640.trt \
# 		--buildOnly \
# 		--fp16


# echo ""
# echo ""
# echo "************************ compile PPSeg model ***************************"
# echo ""
# trtexec --onnx=./onnx_models/mobileseg_mbn3.onnx \
# 		--saveEngine=./mobileseg_mbn3.trt \
# 		--buildOnly \
# 		--fp16

# echo ""
# trtexec --onnx=./onnx_models/ppliteseg_stdc2.onnx \
# 		--saveEngine=./ppliteseg_stdc2.trt \
# 		--buildOnly \
# 		--fp16



# echo ""
# echo ""
# echo "************************ compile MonoDepth model ***************************"
# echo ""
# trtexec --onnx=./onnx_models/litemono-b.onnx \
# 		--saveEngine=./litemono-b.trt \
# 		--buildOnly \
# 		--fp16
