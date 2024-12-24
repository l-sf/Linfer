#include <fstream>
#include <opencv2/opencv.hpp>
#include "yolop/yolop.hpp"
#include <filesystem>

namespace fs = std::filesystem;

using namespace std;

inline string get_file_name(const string &path, bool include_suffix)
{
    if (path.empty())
        return "";
    int p = path.rfind('/');
    int e = path.rfind('\\');
    p = std::max(p, e);
    p += 1;
    // include suffix
    if (include_suffix)
        return path.substr(p);
    int u = path.rfind('.');
    if (u == -1)
        return path.substr(p);

    if (u <= p)
        u = path.size();
    return path.substr(p, u - p);
}

void performance_yolop(const string &engine_file, YoloP::Type type, int gpuid, const string &input_dir)
{
    auto detector = YoloP::create_detector(engine_file, type, gpuid, 0.4, 0.5);
    if (detector == nullptr)
    {
        printf("detector is nullptr.\n");
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

    YoloP::PTMM res;

    // warmup
    for (int i = 0; i < 10; ++i)
        res = detector->detect(images[i % images.size()]);

    // 测试 100 轮
    const int ntest = 100;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < ntest; ++i)
        res = detector->detect(images[i % images.size()]);

    std::chrono::duration<double> during = std::chrono::steady_clock::now() - start;
    double all_time = 1000.0 * during.count();
    float avg_time = all_time / ntest / images.size();
    printf("Average time for %s: %.2f ms, FPS: %.2f\n", engine_file.c_str(), avg_time, 1000 / avg_time);
}

void inference_yolop(const string &engine_file, YoloP::Type type, int gpuid, const string &input_img, const string &output_dir)
{
    auto detector = YoloP::create_detector(engine_file, type, gpuid, 0.4, 0.5);
    if (detector == nullptr)
    {
        printf("detector is nullptr.\n");
        return;
    }
    auto image = cv::imread(input_img);
    if (image.empty())
    {
        printf("Error reading image file: %s\n", input_img.c_str());
        return;
    }
    auto res = detector->detect(image);
    YoloP::BoxArray &boxes = get<0>(res);
    cv::Mat &drive_mask = get<1>(res);
    cv::Mat &lane_mask = get<2>(res);

    for (auto &ibox : boxes)
        cv::rectangle(image, cv::Point(ibox.left, ibox.top),
                      cv::Point(ibox.right, ibox.bottom),
                      {0, 0, 255}, 2);
    fs::create_directories(output_dir);

    string file_name = get_file_name(input_img, false);
    string save_img_path = cv::format("%s/%s.jpg", output_dir.c_str(), file_name.c_str());
    string save_drive_path = cv::format("%s/drive_%s.jpg", output_dir.c_str(), file_name.c_str());
    string save_lane_path = cv::format("%s/lane_%s.jpg", output_dir.c_str(), file_name.c_str());

    cv::imwrite(save_img_path, image);
    cv::imwrite(save_drive_path, drive_mask);
    cv::imwrite(save_lane_path, lane_mask);
    printf("Save to %s, %s, %s\n", save_img_path.c_str(), save_drive_path.c_str(), save_lane_path.c_str());
}