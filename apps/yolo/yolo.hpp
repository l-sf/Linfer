#ifndef YOLO_HPP
#define YOLO_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include "trt_common/trt_tensor.hpp"


/// -------------------------- 封装接口类 ---------------------------
/// -------------------- 支持 YoloV5/X/V7/V8 -----------------------

namespace Yolo{

    using namespace std;

    struct Box{
        float left, top, right, bottom, confidence;
        int label;

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence, int label)
                :left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
    };

    typedef std::vector<Box> BoxArray;

    enum class Type : int{
        V5 = 0,
        X  = 1,
        V7 = 2,
        V8 = 3
    };

    enum class NMSMethod : int{
        CPU = 0, 
        CUDA = 1 
    };

    const char* type_name(Type type);

    class Infer{
    public:
        virtual shared_future<BoxArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, Type type, int gpuid,
        float confidence_threshold=0.25f, float nms_threshold=0.45f,
        NMSMethod nms_method = NMSMethod::CUDA, int max_objects = 1024,
        bool use_multi_preprocess_stream = false
    );

} // namespace Yolo

#endif // YOLO_HPP