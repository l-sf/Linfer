#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "ppseg/ppseg.hpp"
#include <filesystem>

namespace fs = std::filesystem;

using namespace std;

void performance_seg(const string &engine_file, int gpuid, const string &input_dir)
{
    auto predictor = PPSeg::create_seg(engine_file, gpuid);
    if (predictor == nullptr)
    {
        printf("predictor is nullptr.\n");
        return;
    }

    vector<cv::String> files_;
    files_.reserve(100);
    cv::glob(input_dir + "/*.jpg", files_, true);
    vector<string> files(files_.begin(), files_.end());
    if (files.empty())
    {
        printf("No image files found in the input directory: %s\n", input_dir.c_str());
        return;
    }
    std::vector<cv::Mat> images;
    for (const auto &file : files)
    {
        auto image = cv::imread(file);
        if (image.empty())
        {
            printf("Error reading image file: %s\n", file.c_str());
            continue;
        }
        images.emplace_back(image);
    }

    if (images.empty())
    {
        printf("No valid images to process after reading files from: %s\n", input_dir.c_str());
        return;
    }
    int batch = 8;
    for (int i = images.size(); i < batch; ++i)
        images.push_back(images[i % images.size()]);

    // warmup
    cv::Mat res;
    for (int i = 0; i < 10; ++i)
        res = predictor->seg(images[i % images.size()]);

    // 测试 100 轮
    const int ntest = 100;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < ntest; ++i)
        res = predictor->seg(images[i % images.size()]);

    std::chrono::duration<double> during = std::chrono::steady_clock::now() - start;
    double all_time = 1000.0 * during.count();
    float avg_time = all_time / ntest / images.size();
    printf("Average time for %s: %.2f ms, FPS: %.2f\n", engine_file.c_str(), avg_time, 1000 / avg_time);
}

void inference_seg(const string &engine_file, int gpuid, const string &input_img, const string &output_img_path)
{
    auto predictor = PPSeg::create_seg(engine_file, gpuid);
    if (predictor == nullptr)
    {
        printf("predictor is nullptr.\n");
        return;
    }

    auto image = cv::imread(input_img);
    if (image.empty())
    {
        printf("Error reading image file: %s\n", input_img.c_str());
        return;
    }
    auto res = predictor->seg(image);

    cv::Mat color_img(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    // 遍历每个像素点，根据类别索引应用颜色映射
    for (int i = 0; i < res.rows; ++i)
    {
        for (int j = 0; j < res.cols; ++j)
        {
            uchar pixel_value = res.at<uchar>(i, j);
            color_img.at<cv::Vec3b>(i, j) = cv::Vec3b(PPSeg::color_map[pixel_value][2],
                                                      PPSeg::color_map[pixel_value][1],
                                                      PPSeg::color_map[pixel_value][0]);
        }
    }
    cv::Mat out_color_img(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    float alpha = 0.7;
    out_color_img = (1 - alpha) * image + alpha * color_img;
    fs::path path(output_img_path);
    fs::create_directories(path.parent_path());
    cv::imwrite(output_img_path, out_color_img);
    printf("Save to %s\n", output_img_path.c_str());
}