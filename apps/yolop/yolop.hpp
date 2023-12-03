
#ifndef YOLOP_HPP
#define YOLOP_HPP


#include <vector>
#include <tuple>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include "trt_common/trt_tensor.hpp"


namespace YoloP{

    using namespace std;

    struct Box{
        float left, top, right, bottom, confidence;
        int label;

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence, int label)
                :left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
    };

    using BoxArray = std::vector<Box>;
    // detect函数的返回值类型
    using PTMM = std::tuple<BoxArray, cv::Mat, cv::Mat>;

    enum class Type : int{
        V1 = 0,
        V2 = 1,
    };

    enum class NMSMethod : int{
        CPU = 0,
        CUDA = 1
    };

    class Detector{
    public:
        virtual PTMM detect(const cv::Mat& image) = 0;
    };

    shared_ptr<Detector> create_detector(
            const string& engine_file, Type type, int gpuid = 0,
            float confidence_threshold=0.4f, float nms_threshold=0.5f,
            NMSMethod nms_method = NMSMethod::CUDA, int max_objects = 512
    );

} // namespace YoloP

#endif //YOLOP_HPP
