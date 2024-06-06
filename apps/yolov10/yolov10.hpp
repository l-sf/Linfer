#ifndef LINFER_YOLOV10_HPP
#define LINFER_YOLOV10_HPP


#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>


namespace YOLOV10{

    using namespace std;

    struct Box{
        float left, top, right, bottom, confidence;
        int label;

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence, int label)
                :left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
    };

    using BoxArray = std::vector<Box>;

    class Infer{
    public:
        virtual shared_future<BoxArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(
            const string& engine_file, int gpuid,
            float confidence_threshold = 0.5f, int max_objects = 300,
            bool use_multi_preprocess_stream = false
    );


} // namespace YOLOV10


#endif //LINFER_YOLOV10_HPP
