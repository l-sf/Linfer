

#ifndef DEPLOY_YOLOV5_HPP
#define DEPLOY_YOLOV5_HPP

#include <future>
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "trt_common/trt-tensor.hpp"


/// -------------------------- 封装接口类 ---------------------------

namespace Yolo {

    struct Box{
        float left, top, right, bottom, confidence;
        int label;
        Box() = default;
        Box(float left, float top, float right, float bottom, float confidence, int label)
            : left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label) {}
    };

    using BoxArray = std::vector<Box>;

    enum class Type : int{
        V5 = 0,
        X = 1,
    };

    enum class NMSMethod : int{
        CPU = 0,
        FastGPU = 1
    };

    class Infer{
    public:
        virtual std::shared_future<BoxArray> commit(const cv::Mat& input) = 0;
        virtual std::vector<std::shared_future<BoxArray>> commits(const std::vector<cv::Mat>& images) = 0;
    };

    std::shared_ptr<Infer> create_infer(const std::string& file, Type type, int gpuid = 0,
                                        float confidence_threshold = 0.25, float nms_threshole = 0.45,
                                        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
                                        bool use_multi_preprocess_stream = false
                                        );

}

#endif //DEPLOY_YOLOV5_HPP
