
#include <fstream>
#include <opencv2/opencv.hpp>
#include "yolop/yolop.hpp"

using namespace std;

void performance_yolop(const string& engine_file, YoloP::Type type, int gpuid){
    auto detector = YoloP::create_detector(engine_file, type, gpuid, 0.4, 0.5);
    if(detector == nullptr){
        printf("detector is nullptr.\n");
        return;
    }

    auto image = cv::imread("imgs/1.jpg");
    YoloP::PTMM res;

    // warmup
    for(int i = 0; i < 10; ++i)
        res = detector->detect(image);

    // 测试 100 轮
    const int ntest = 100;
    auto start = std::chrono::steady_clock::now();
    for(int i  = 0; i < ntest; ++i)
        res = detector->detect(image);

    std::chrono::duration<double> during = std::chrono::steady_clock::now() - start;
    double all_time = 1000.0 * during.count();
    float avg_time = all_time / ntest;
    printf("Average time: %.2f ms, FPS: %.2f\n", engine_file.c_str(), avg_time, 1000 / avg_time);
}


void inference_yolop(const string& engine_file, YoloP::Type type, int gpuid){
    auto detector = YoloP::create_detector(engine_file, type, gpuid, 0.4, 0.5);
    if(detector == nullptr){
        printf("detector is nullptr.\n");
        return;
    }

    auto image = cv::imread("imgs/6.jpg");
    auto res = detector->detect(image);
    YoloP::BoxArray& boxes = get<0>(res);
    cv::Mat& drive_mask = get<1>(res);
    cv::Mat& lane_mask = get<2>(res);

    for(auto& ibox : boxes)
        cv::rectangle(image, cv::Point(ibox.left, ibox.top),
                      cv::Point(ibox.right, ibox.bottom),
                      {0, 0, 255}, 2);

    cv::imwrite("infer_res/res.jpg", image);
    cv::imwrite("infer_res/drive.jpg", drive_mask);
    cv::imwrite("infer_res/lane.jpg", lane_mask);
}

