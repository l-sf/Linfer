
#include <fstream>
#include <opencv2/opencv.hpp>
#include "ppseg/ppseg.hpp"

using namespace std;

void performance_seg(const string& engine_file, int gpuid){
    auto predictor = PPSeg::create_seg(engine_file, gpuid);
    if(predictor == nullptr){
        printf("predictor is nullptr.\n");
        return;
    }

    auto image = cv::imread("imgs/frame_0.jpg");

    cv::Mat res;
    // warmup
    for(int i = 0; i < 10; ++i)
        res = predictor->seg(image);

    // 测试 100 轮
    const int ntest = 100;
    auto start = std::chrono::steady_clock::now();
    for(int i  = 0; i < ntest; ++i)
        res = predictor->seg(image);

    std::chrono::duration<double> during = std::chrono::steady_clock::now() - start;
    double all_time = 1000.0 * during.count();
    float avg_time = all_time / ntest;
    printf("Average time: %.2f ms, FPS: %.2f\n", engine_file.c_str(), avg_time, 1000 / avg_time);
}

void inference_seg(const string& engine_file, int gpuid){
    auto predictor = PPSeg::create_seg(engine_file, gpuid);
    if(predictor == nullptr){
        printf("predictor is nullptr.\n");
        return;
    }

    auto image = cv::imread("imgs/frame_3.jpg");
    auto res = predictor->seg(image);

    cv::Mat color_img(res.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    // 遍历每个像素点，根据类别索引应用颜色映射
    for (int i = 0; i < res.rows; ++i) {
        for (int j = 0; j < res.cols; ++j) {
            uchar pixel_value = res.at<uchar>(i, j);
            color_img.at<cv::Vec3b>(i, j) = cv::Vec3b(PPSeg::color_map[pixel_value][2],
                                                          PPSeg::color_map[pixel_value][1],
                                                          PPSeg::color_map[pixel_value][0]);
        }
    }
    cv::Mat out_color_img;
    cv::resize(color_img, out_color_img, image.size());
    out_color_img.convertTo(out_color_img, CV_8UC3);
    float alpha = 0.7;
    out_color_img = (1 - alpha) * image + alpha * out_color_img;
    cv::imwrite("infer_res/seg_frame_3.jpg", out_color_img);
}



