
#include <opencv2/opencv.hpp>
#include "apps/yolo/yolo.hpp"
#include "apps/yolop/yolop.hpp"

using namespace std;

void performance_v10(const string& engine_file, int gpuid);
void batch_inference_v10(const string& engine_file, int gpuid);
void single_inference_v10(const string& engine_file, int gpuid);
void performance(const string& engine_file, int gpuid);
void batch_inference(const string& engine_file, int gpuid);
void single_inference(const string& engine_file, int gpuid);
void performance(const string& engine_file, int gpuid, Yolo::Type type);
void batch_inference(const string& engine_file, int gpuid, Yolo::Type type);
void single_inference(const string& engine_file, int gpuid, Yolo::Type type);
void inference_bytetrack(const string& engine_file, int gpuid, Yolo::Type type, const string& video_file);
void infer_track(int Mode, const string& path);
void inference_yolop(const string& engine_file, YoloP::Type type, int gpuid);
void performance_yolop(const string& engine_file, YoloP::Type type, int gpuid);
void inference_seg(const string& engine_file, int gpuid);
void performance_seg(const string& engine_file, int gpuid);
bool test_ptq();

void test_rtdetr(){
//    batch_inference("rtdetr_r50vd_6x_coco_dynamic_fp16.trt", 0);
//    single_inference("rtdetr_r50vd_6x_coco_dynamic_fp16.trt", 0);
    performance("rtdetr_r50vd_6x_coco_dynamic_fp16.trt", 0);
}

void test_yolov10(){
//    batch_inference_v10("yolov10l.trt", 0);
//    single_inference_v10("yolov10l.trt", 0);
    performance_v10("yolov10n.trt", 0);
}

void test_yolo(){
//    batch_inference("yolov5s.trt", 0, Yolo::Type::V5);
    performance("yolov5s.trt", 0, Yolo::Type::V5);
//    batch_inference("yolov5s_ptq.trt", 0, Yolo::Type::V5);
//    batch_inference("yolov5m.trt", 0, Yolo::Type::V5);
//    performance("yolov5m.trt", 0, Yolo::Type::V5);
//    batch_inference("yolox_s.trt", 0, Yolo::Type::X);
//    performance("yolox_s.trt", 0, Yolo::Type::X);
//    batch_inference("yolox_m.trt", 0, Yolo::Type::X);
//    performance("yolox_m.trt", 0, Yolo::Type::X);
//    batch_inference("yolov7.trt", 0, Yolo::Type::V7);
//    performance("yolov7.trt", 0, Yolo::Type::V7);
//    batch_inference("yolov7_qat.trt", 0, Yolo::Type::V7);
//    performance("yolov7_qat.trt", 0, Yolo::Type::V7);
//    batch_inference("yolov8n.trt", 0, Yolo::Type::V8);
//    performance("yolov8n.trt", 0, Yolo::Type::V8);
//    batch_inference("yolov8s.trt", 0, Yolo::Type::V8);
//    performance("yolov8s.trt", 0, Yolo::Type::V8);
//    batch_inference("yolov8l.trt", 0, Yolo::Type::V8);
//    performance("yolov8l.trt", 0, Yolo::Type::V8);
//    single_inference("yolov8l.trt", 0, Yolo::Type::V8);
}

void test_track(){
//    inference_bytetrack("yolov8s.trt", 0, Yolo::Type::V8, "videos/palace.mp4");
    infer_track(2, "Woman/img/%04d.jpg");
}

void test_yolop(){
    inference_yolop("yolopv2-480x640.trt", YoloP::Type::V2, 0);
//    inference_yolop("yolop-640.trt", YoloP::Type::V1, 0);
//    performance_yolop("yolopv2-480x640.trt", YoloP::Type::V2, 0);
//    performance_yolop("yolop-640.trt", YoloP::Type::V1, 0);
}

void test_seg(){
    inference_seg("ppliteseg_stdc2.trt", 0);
//    inference_seg("mobileseg_mbn3.trt", 0);
//    performance_seg("ppliteseg_stdc2.trt", 0);
//    performance_seg("mobileseg_mbn3.trt", 0);
}

int main(){
//    test_rtdetr();
    test_yolov10();
//    test_yolo();
//    test_yolop();
//    test_track();
//    test_ptq();
    test_seg();
    return 0;
}