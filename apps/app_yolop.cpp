
#include <fstream>
#include <opencv2/opencv.hpp>
#include "yolop/yolop.hpp"

using namespace std;


void performance_yolop(const string& engine_file, int gpuid){
    auto detector = YoloP::create_detector(engine_file, gpuid, 0.25, 0.45);
    if(detector == nullptr){
        printf("detector is nullptr.\n");
        return;
    }

    auto image = cv::imread("imgs/1.jpg");
    YoloP::PBM res;

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


void inference_yolop(const string& engine_file, int gpuid){
    auto detector = YoloP::create_detector(engine_file, gpuid, 0.3, 0.45);
    if(detector == nullptr){
        printf("detector is nullptr.\n");
        return;
    }

    auto image = cv::imread("imgs/6.jpg");
    auto res = detector->detect(image);
    YoloP::BoxArray& boxes = res.first;
    cv::Mat& out_img = res.second;
    for(auto& ibox : boxes){
        cv::Scalar color(0, 0, 255);
        cv::rectangle(out_img, cv::Point(ibox.left, ibox.top), cv::Point(ibox.right, ibox.bottom), color, 2);
        string name = "car";
        auto caption = cv::format("%s %.2f", name.c_str(), ibox.confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(out_img, cv::Point(ibox.left-2, ibox.top-32), cv::Point(ibox.left + text_width, ibox.top), color, -1);
        cv::putText(out_img, caption, cv::Point(ibox.left, ibox.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imwrite("infer_res/yolop_res6.jpg", out_img);
}

void infer_video_yolop(const string& engine_file, int gpuid){
    auto detector = YoloP::create_detector(engine_file, gpuid, 0.4, 0.45);
    if(detector == nullptr){
        printf("detector is nullptr.\n");
        return;
    }
    cv::VideoCapture cap;
    cap.open("videos/road.mp4");
    auto fps = cap.get(cv::CAP_PROP_FPS);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter writer("videos/res_yolop.mp4", cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), fps, cv::Size(width, height));
    cv::Mat image;
    while (true){
        bool ret = cap.read(image);
        if (!ret) {
            cout << "----------Read failed!!!----------" << endl;
            return;
        }
        auto start = std::chrono::steady_clock::now();
        auto res = detector->detect(image);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double time = 1000 * elapsed.count();
        printf("all infer time: %f ms\n", time);
        YoloP::BoxArray& boxes = res.first;
        cv::Mat& out_img = res.second;
        for(auto& ibox : boxes){
            cv::Scalar color(0, 0, 255);
            cv::rectangle(out_img, cv::Point(ibox.left, ibox.top), cv::Point(ibox.right, ibox.bottom), color, 2);
            string name = "car";
            auto caption = cv::format("%s %.2f", name.c_str(), ibox.confidence);
            int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(out_img, cv::Point(ibox.left-2, ibox.top-32), cv::Point(ibox.left + text_width, ibox.top), color, -1);
            cv::putText(out_img, caption, cv::Point(ibox.left, ibox.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }
        writer.write(out_img);
        cv::imshow("Yolop", out_img);
        cv::waitKey(1);
    }
}

