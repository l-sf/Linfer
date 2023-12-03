
#include <fstream>
#include <opencv2/opencv.hpp>
#include "yolop/yolop.hpp"

using namespace std;

inline vector<string> cocolabels = {
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
};

inline std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
        case 0:r = v; g = t; b = p;break;
        case 1:r = q; g = v; b = p;break;
        case 2:r = p; g = v; b = t;break;
        case 3:r = p; g = q; b = v;break;
        case 4:r = t; g = p; b = v;break;
        case 5:r = v; g = p; b = q;break;
        default:r = 1; g = 1; b = 1;break;}
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

inline std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

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

