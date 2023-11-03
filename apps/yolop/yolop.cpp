
#include "yolop.hpp"
#include <condition_variable>
#include "trt_common/trt_infer.hpp"
#include "trt_common/ilogger.hpp"
#include "trt_common/preprocess_kernel.cuh"
#include "trt_common/cuda_tools.hpp"

namespace YoloP{

    using namespace std;

    void decode_box_kernel_invoker(
            float* predict, int num_bboxes, int num_classes, float confidence_threshold,
            float* invert_affine_matrix, float* parray,
            int max_objects, cudaStream_t stream
    );

    void decode_drive_lane_kernel_invoker(
            float* pred_drive, float* parray_drive,
            float* pred_lane, float* parray_lane,
            int height, int width, cudaStream_t stream
    );

    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;
            float scale = std::min(scale_x, scale_y);

            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * from.width + to.width + scale - 1) * 0.5f;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * from.height + to.height + scale - 1) * 0.5f;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat d2i_mat(){
            return {2, 3, CV_32F, d2i};
        }
    };

    static float iou(const Box& a, const Box& b){
        float cross_left   = std::max(a.left, b.left);
        float cross_top    = std::max(a.top, b.top);
        float cross_right  = std::min(a.right, b.right);
        float cross_bottom = std::min(a.bottom, b.bottom);

        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        float union_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top)
                           + std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) - cross_area;
        if(cross_area == 0.f || union_area == 0.f) return 0.0f;
        return cross_area / union_area;
    }

    static BoxArray cpu_nms(BoxArray& boxes, float threshold){
        std::sort(boxes.begin(), boxes.end(), [](Box& a, Box& b){return a.confidence > b.confidence;});
        BoxArray box_result(boxes.size());
        vector<bool> remove_flags(boxes.size());
        for(int i = 0; i < boxes.size(); ++i){
            if(remove_flags[i]) continue;
            auto& a = boxes[i];
            box_result.emplace_back(a);
            for(int j = i + 1; j < boxes.size(); ++j){
                if(remove_flags[j]) continue;
                auto& b = boxes[j];
                if(iou(a, b) >= threshold)
                    remove_flags[j] = true;
            }
        }
        return box_result;
    }

    class DetectorImpl : public Detector{
    public:
        ~DetectorImpl() = default;

        bool startup(const string& file, int gpuid, float confidence_threshold,
                     float nms_threshold, int max_objects
        ){
            normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.f, 0.f, CUDAKernel::ChannelType::Invert);
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;
            max_objects_          = max_objects;

            TRT::set_device(gpuid);
            model_ = TRT::load_infer(file);
            if(model_ == nullptr){
                INFOE("Load model failed: %s", file.c_str());
                return false;
            }

            model_->print();

            // 绑定输入输出
            input_ = model_->input();
            det_out_ = model_->output(0);
            drive_seg_ = model_->output(1);
            lane_seg_ = model_->output(2);
            input_width_  = input_->shape(3);
            input_height_ = input_->shape(2);
            stream_       = model_->get_stream();
            gpu_          = gpuid;

            // 分配 input Tensor GPU 内存
            input_->to_gpu();
            // 这里的 1 + MAX_IMAGE_BBOX 结构是，counter + bboxes ...
            // 先调整shape再分配GPU内存
            det_array_->resize(1, 1 + max_objects_ * num_box_elements_).to_gpu();
            drive_array_->resize(1, input_height_, input_width_).to_gpu();
            lane_array_->resize(1, input_height_, input_width_).to_gpu();

            return true;
        }

        shared_ptr<TBMM> detect(const cv::Mat& image) override{
            // 预处理
            AffineMatrix affineMatrix{};
            preprocess(image, affineMatrix);

            // 推理
            model_->forward(false);

            // 后处理
            checkCudaRuntime(cudaMemsetAsync(det_array_->gpu<float>(), 0, sizeof(int), stream_));
            decode_box_kernel_invoker(det_out_->gpu<float>(), det_out_->shape(1), num_classes_, confidence_threshold_,
                                  affineMatrix.d2i, det_array_->gpu<float>(), max_objects_, stream_);

            decode_drive_lane_kernel_invoker(drive_seg_->gpu<float>(), drive_array_->gpu<float>(),
                                    lane_seg_->gpu<float>(), lane_array_->gpu<float>(),
                                        input_height_, input_width_, stream_);
            model_->synchronize();
            stream_ = nullptr;
            det_array_->to_cpu();
            drive_array_->to_cpu();
            lane_array_->to_cpu();

            auto* parray = det_array_->cpu<float>();
            int count = min(max_objects_, (int)*parray);
            BoxArray image_based_boxes(count);
            for(int i = 0; i < count; ++i){
                float* pbox  = parray + 1 + i * num_box_elements_;
                int keepflag = pbox[5];
                if(keepflag == 1){
                    image_based_boxes.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4]);
                }
            }

            image_based_boxes = cpu_nms(image_based_boxes, nms_threshold_);

//            cv::Mat drive_mask_tmp(input_height_, input_width_, CV_32FC1, drive_array_->cpu<float>());
//            cv::Mat lane_mask_tmp(input_height_, input_width_, CV_32FC1, drive_array_->cpu<float>());
//            cv::Mat drive_mask;
//            cv::Mat lane_mask;
//            cv::Mat m2x3_d2i = affineMatrix.d2i_mat();  // dst to image, 2x3 matrix
//            cv::warpAffine(drive_mask_tmp, drive_mask, m2x3_d2i, image.size(),
//                           cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        }

        bool preprocess(const cv::Mat& image, AffineMatrix& affineMatrix) {
            if(image.empty()){
                INFOE("Image is empty.");
                return false;
            }
            auto tensor = input_;
            if(tensor == nullptr){
                INFOE("Input Tensor is empty.");
                return false;
            }
            TRT::CUStream preprocess_stream = tensor->get_stream();

            cv::Size input_size(input_width_, input_height_);
            affineMatrix.compute(image.size(), input_size);

            tensor->resize(1, 3, input_height_, input_width_);

            size_t size_image = image.cols * image.rows * 3;
            // 对齐 32 字节
            size_t size_matrix = iLogger::upbound(sizeof(affineMatrix.d2i), 32);
            auto workspace = tensor->get_workspace();
            auto* gpu_workspace           = (uint8_t*)workspace->gpu(size_matrix + size_image);
            auto*   affine_matrix_device  = (float*)gpu_workspace;
            uint8_t* image_device         = size_matrix + gpu_workspace;

            auto* cpu_workspace           = (uint8_t*)workspace->cpu(size_matrix + size_image);
            auto* affine_matrix_host      = (float*)cpu_workspace;
            uint8_t* image_host           = size_matrix + cpu_workspace;

            // speed up
            memcpy(image_host, image.data, size_image);
            memcpy(affine_matrix_host, affineMatrix.d2i, sizeof(affineMatrix.d2i));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affineMatrix.d2i), cudaMemcpyHostToDevice, preprocess_stream));

            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                    image_device, image.cols * 3, image.cols, image.rows,
                    tensor->gpu<float>(), input_width_, input_height_,
                    affine_matrix_device, 114,
                    normalize_, preprocess_stream
            );

            return true;
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_        = 0;
        int num_classes_            = 1;
        int max_objects_            = 512;
        int num_box_elements_       = 6;  // left, top, right, bottom, confidence, keepflag

        TRT::CUStream stream_       = nullptr;
        CUDAKernel::Norm normalize_;

        shared_ptr<TRT::Infer> model_;
        shared_ptr<TRT::Tensor> input_;
        shared_ptr<TRT::Tensor> det_out_;
        shared_ptr<TRT::Tensor> drive_seg_;
        shared_ptr<TRT::Tensor> lane_seg_;

        // 这三个 是输出经过 decode 的结果
        shared_ptr<TRT::Tensor> det_array_;
        shared_ptr<TRT::Tensor> drive_array_;
        shared_ptr<TRT::Tensor> lane_array_;
    };

    shared_ptr<Detector> create_detector(
            const string& engine_file, int gpuid,
            float confidence_threshold, float nms_threshold, int max_objects
    ){
        shared_ptr<DetectorImpl> instance(new DetectorImpl{});
        if(!instance->startup(engine_file, gpuid, confidence_threshold, nms_threshold, max_objects)){
            instance.reset();
        }
        return instance;
    }

}