
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

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence)
                :left(left), top(top), right(right), bottom(bottom), confidence(confidence){}
    };

    using BoxArray = std::vector<Box>;
    using PTMM = std::tuple<BoxArray, cv::Mat, cv::Mat>;


    class Detector{
    public:
        virtual PTMM detect(const cv::Mat& image) = 0;
    };

    shared_ptr<Detector> create_detector(
            const string& engine_file, int gpuid = 0,
            float confidence_threshold=0.3f, float nms_threshold=0.45f,
            int max_objects = 256
    );

} // namespace YoloP

#endif //YOLOP_HPP
