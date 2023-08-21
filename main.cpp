
#include <opencv2/opencv.hpp>
#include "apps/yolo/yolo.hpp"

using namespace std;

void performance(const string& engine_file, int gpuid, Yolo::Type type);
void batch_inference(const string& engine_file, int gpuid, Yolo::Type type);
void single_inference(const string& engine_file, int gpuid, Yolo::Type type);
void inference_bytetrack(const string& engine_file, int gpuid, Yolo::Type type, const string& video_file);


void test_yolo(){
//    batch_inference("yolov5s.trt", 0, Yolo::Type::V5);
//    performance("yolov5s.trt", 0, Yolo::Type::V5);
//    batch_inference("yolov5s_ptq.trt", 0, Yolo::Type::V5);
//    performance("yolov5s_ptq.trt", 0, Yolo::Type::V5);
//    batch_inference("yolov5m.trt", 0, Yolo::Type::V5);
//    performance("yolov5m.trt", 0, Yolo::Type::V5);
//    batch_inference("yolov5m_ptq.trt", 0, Yolo::Type::V5);
//    performance("yolov5m_ptq.trt", 0, Yolo::Type::V5);
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
    single_inference("yolov8s.trt", 0, Yolo::Type::V8);
    batch_inference("yolov8s.trt", 0, Yolo::Type::V8);
    performance("yolov8s.trt", 0, Yolo::Type::V8);
//    batch_inference("yolov8m.trt", 0, Yolo::Type::V8);
//    performance("yolov8m.trt", 0, Yolo::Type::V8);
}

void test_track(){
    inference_bytetrack("yolov8s.trt", 0, Yolo::Type::V8, "videos/palace.mp4");
}

int main(){
    test_yolo();
//    test_track();
    return 0;
}