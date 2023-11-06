
#include <fstream>
#include <opencv2/opencv.hpp>
#include "yolo/yolo.hpp"

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

inline string get_file_name(const string& path, bool include_suffix){
    if (path.empty()) return "";
    int p = path.rfind('/');
    int e = path.rfind('\\');
    p = std::max(p, e);
    p += 1;
    //include suffix
    if (include_suffix)
        return path.substr(p);
    int u = path.rfind('.');
    if (u == -1)
        return path.substr(p);

    if (u <= p) u = path.size();
    return path.substr(p, u - p);
}


void performance(const string& engine_file, int gpuid, Yolo::Type type){
    auto infer = Yolo::create_infer(engine_file, type, gpuid, 0.3, 0.45);
    if(infer == nullptr){
        printf("infer is nullptr.\n");
        return;
    }

    int batch = 8;
    std::vector<cv::Mat> images{cv::imread("imgs/bus.jpg"), cv::imread("imgs/girl.jpg"),
                                cv::imread("imgs/group.jpg"), cv::imread("imgs/yq.jpg")};
    for (int i = images.size(); i < batch; ++i)
        images.push_back(images[i % 4]);

    // warmup
    vector<shared_future<Yolo::BoxArray>> boxes_array;
    for(int i = 0; i < 10; ++i)
        boxes_array = infer->commits(images);
    boxes_array.back().get();
    boxes_array.clear();

    // 测试 100 轮
    const int ntest = 100;
    auto start = std::chrono::steady_clock::now();
    for(int i  = 0; i < ntest; ++i)
        boxes_array = infer->commits(images);
    // 等待全部推理结束
    boxes_array.back().get();

    std::chrono::duration<double> during = std::chrono::steady_clock::now() - start;
    double all_time = 1000.0 * during.count();
    float avg_time = all_time / ntest / images.size();
    printf("Average time: %.2f ms, FPS: %.2f\n", engine_file.c_str(), avg_time, 1000 / avg_time);
}


void batch_inference(const string& engine_file, int gpuid, Yolo::Type type){
    auto infer = Yolo::create_infer(engine_file, type, gpuid, 0.25, 0.45);
    if(infer == nullptr){
        printf("infer is nullptr.\n");
        return;
    }

    vector<cv::String> files_;
    files_.reserve(100);
    cv::glob("imgs/*.jpg", files_, true);
    vector<string> files(files_.begin(), files_.end());

    vector<cv::Mat> images;
    for(const auto& file : files){
        auto image = cv::imread(file);
        images.emplace_back(image);
    }

    vector<shared_future<Yolo::BoxArray>> boxes_array;
    boxes_array = infer->commits(images);

    // 等待全部推理结束
    boxes_array.back().get();

    string root_res = "infer_res";
    for(int i = 0; i < boxes_array.size(); ++i){
        cv::Mat image = images[i];
        auto boxes = boxes_array[i].get();
        for(auto & ibox : boxes){
            cv::Scalar color;
            std::tie(color[0], color[1], color[2]) = random_color(ibox.label);
            cv::rectangle(image, cv::Point(ibox.left, ibox.top), cv::Point(ibox.right, ibox.bottom), color, 2);

            auto name = cocolabels[ibox.label];
            auto caption = cv::format("%s %.2f", name.c_str(), ibox.confidence);
            int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(ibox.left-2, ibox.top-32), cv::Point(ibox.left + text_width, ibox.top), color, -1);
            cv::putText(image, caption, cv::Point(ibox.left, ibox.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }
        string file_name = get_file_name(files[i], false);
        string save_path = cv::format("%s/%s.jpg", root_res.c_str(), file_name.c_str());
        cv::imwrite(save_path, image);
        printf("Save to %s, %d object\n", save_path.c_str(), boxes.size());
    }
}

void single_inference(const string& engine_file, int gpuid, Yolo::Type type){
    auto infer = Yolo::create_infer(engine_file, type, gpuid, 0.25, 0.45);
    if(infer == nullptr){
        printf("infer is nullptr.\n");
        return;
    }

    auto image = cv::imread("imgs/bus.jpg");
    auto boxes = infer->commit(image).get();
    for(auto& ibox : boxes){
        cv::Scalar color;
        std::tie(color[0], color[1], color[2]) = random_color(ibox.label);
        cv::rectangle(image, cv::Point(ibox.left, ibox.top), cv::Point(ibox.right, ibox.bottom), color, 2);

        auto name = cocolabels[ibox.label];
        auto caption = cv::format("%s %.2f", name.c_str(), ibox.confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(ibox.left-2, ibox.top-32), cv::Point(ibox.left + text_width, ibox.top), color, -1);
        cv::putText(image, caption, cv::Point(ibox.left, ibox.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imwrite("infer_res/result.jpg", image);
}

