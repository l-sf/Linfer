

#include "yolov5.hpp"
#include <condition_variable>
#include <future>
#include <vector>
#include <string>
#include <functional>
#include "tensorrt/cuda-tools.hpp"
#include "tensorrt/ilogger.hpp"
#include "tensorrt/trt-infer.hpp"
#include "tensorrt/producer_consumer.hpp"
#include "tensorrt/preprocess_kernel.cuh"


using namespace std;

namespace Yolo{

    void decode_kernel_invoker(
        float* predict, int num_boxes, int num_classes, float confidence_threshold,
        float* invert_affine_matrix, float* parray,
        int max_objects, cudaStream_t stream);

    void nms_kernel_invoker(
            float* parray, float nms_threshold, int max_objects, cudaStream_t stream
            );

    struct AffineMatrix{
        float i2d[6];
        float d2i[6];

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = (float)to.width / (float)from.width;
            float scale_y = (float)to.height / (float)from.height;
            float scale = min(scale_x, scale_y);

            i2d[0] = scale; i2d[1] = 0; i2d[2] = (-scale * from.width + to.width + scale - 1) * 0.5f;
            i2d[3] = 0; i2d[4] = scale; i2d[5] = (-scale * from.height + to.height + scale - 1) * 0.5f;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
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
        vector<Box> box_result(boxes.size());
        vector<bool> remove_flags(boxes.size());
        for(int i = 0; i < boxes.size(); ++i){
            if(remove_flags[i]) continue;
            Box ibox = boxes[i];
            for(int j = i + 1; j < boxes.size(); ++j){
                if(remove_flags[j]) continue;
                Box jbox = boxes[j];
                if(ibox.label == jbox.label){
                    if(iou(ibox, jbox) > threshold)
                        remove_flags[j] = true;
                }
            }
        }
        return box_result;
    }

    using ControllerImpl = InferController<cv::Mat, BoxArray, tuple<string, int>, AffineMatrix>;

    class InferImpl : public Infer, public ControllerImpl {
    public:
        ~InferImpl() override { stop(); }

        shared_future<BoxArray> commit(const cv::Mat& image) override {
            return ControllerImpl::commit(image);
        }

        vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) override{
            return ControllerImpl::commits(images);
        }

        bool startup(const string& file, Type type, int gpuid, float confidence_threshold, float nms_threshold,
                     NMSMethod nms_method, int max_objects, bool use_multi_preprocess_stream) {
            if(type == Type::V5){
                normalize_ = CUDAKernel::Norm::alpha_beta(1/255.f, 0.f, CUDAKernel::ChannelType::Invert);
            }
            else if(type == Type::X){
//                float mean[] = {0.485, 0.456, 0.406};
//                float std[]  = {0.229, 0.224, 0.225};
//                normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);
                normalize_ = CUDAKernel::Norm::None();
            }
            else{
                INFOE("Unsupported type %d", type);
            }

            confidence_threshold_ = confidence_threshold;
            nms_threshold_ = nms_threshold;
            nms_method_ = nms_method;
            max_objects_ = max_objects;
            use_multi_preprocess_stream_ = use_multi_preprocess_stream;

            return ControllerImpl::startup(make_tuple(file, gpuid));
        }

        void worker(promise<bool>& pro) override{
            // load model
            string file = std::get<0>(start_param_);
            int gpuid = std::get<1>(start_param_);
            checkCudaRuntime(cudaSetDevice(gpuid));
            auto model = TRT::load_infer(file);
            if(model == nullptr){
                // failed
                pro.set_value(false);
                INFOE("Load model failed: %s", file.c_str());
                return;
            }

            model->print();

            const int MAX_IMAGE_BOX = max_objects_;
            const int NUM_BOX_ELEMENT = 7;  // left, top, right, bottom, confidence, class, keepflag
            auto input = model->input();
            auto output = model->output();
            int max_batch_size = model->get_max_batch_size();
            int num_classes = output->shape(2) - 5;
            input_width_ = input->shape(3);
            input_height_ = input->shape(2);
            tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_ = model->get_stream();
            gpuid_ = gpuid;
            // load success：设置好了输入，宽高，batch，allocator等
            pro.set_value(true);

            TRT::Tensor affine_matrix_device(TRT::DataType::Float);
            TRT::Tensor output_array_device(TRT::DataType::Float);
            affine_matrix_device.set_stream(stream_);
            // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
            affine_matrix_device.resize(max_batch_size, 8).to_gpu();
            // output_array_device 是输出 output 经过 decode 的结果
            // 这里的 1 + MAX_IMAGE_BBOX 结构是，count + bboxes ...
            output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BOX * NUM_BOX_ELEMENT).to_gpu();

            vector<Job> fetched_jobs;
            while(get_jobs_and_wait(fetched_jobs, max_batch_size)){
                int infer_batch_size = fetched_jobs.size();
                input->resize_single_dim(0, infer_batch_size).to_gpu();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job = fetched_jobs[ibatch];
                    auto& mono_tensor = job.mono_tensor->data();
                    if(mono_tensor->get_stream() != stream_)
                        checkCudaRuntime(cudaStreamSynchronize(mono_tensor->get_stream()));
                    affine_matrix_device.copy_from_gpu(affine_matrix_device.offset(ibatch), mono_tensor->get_workspace()->gpu(), 6);
                    input->copy_from_gpu(input->offset(ibatch), mono_tensor->gpu(), mono_tensor->numel());
                    job.mono_tensor->release();
                    INFO("333333333333333333");
                }

                // 进行推理，一次推理一批
                model->forward(false);

                output_array_device.to_gpu(false);
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job = fetched_jobs[ibatch];
                    auto* predict_batch = output->gpu<float>(ibatch);
                    auto* output_array_ptr = output_array_device.gpu<float>(ibatch);
                    auto affine_matrix = affine_matrix_device.gpu<float>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(predict_batch, 0, sizeof(int), stream_));
                    decode_kernel_invoker(predict_batch, output->shape(1), num_classes, confidence_threshold_,
                                          affine_matrix, output_array_ptr, MAX_IMAGE_BOX, stream_);
                    if(nms_method_ == NMSMethod::FastGPU)
                        nms_kernel_invoker(output_array_ptr, nms_threshold_, MAX_IMAGE_BOX, stream_);
                    INFO("4444444444444444444");
                }

                output_array_device.to_cpu();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    float* parray = output_array_device.cpu<float>(ibatch);
                    int count = std::min(MAX_IMAGE_BOX, (int)*parray);
                    auto& job = fetched_jobs[ibatch];
                    auto& image_boxes = job.output;
                    for(int i = 0; i < count; ++i){
                        float* pbox = parray + 1 + i * NUM_BOX_ELEMENT;
                        int label = pbox[5];
                        int keepflag = pbox[6];
                        if(keepflag == 1)
                            image_boxes.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                    }

                    if(nms_method_ == NMSMethod::CPU)
                        image_boxes = cpu_nms(image_boxes, nms_threshold_);

                    job.pro->set_value(image_boxes);
                }
                fetched_jobs.clear();
            }
            INFO("14141414141414");
            stream_ = nullptr;
            INFO("1515151515115");
            tensor_allocator_.reset();
            INFO("Infer worker done.");
        }

        bool preprocess(Job& job, const cv::Mat& image) override {
            if(tensor_allocator_ == nullptr){
                INFOE("Tensor_allocator_ is nullptr.");
                return false;
            }
            if(image.empty()){
                INFOE("Image is empty.");
                return false;
            }

            job.mono_tensor = tensor_allocator_->query();
            if(job.mono_tensor == nullptr){
                INFOE("Tensor allocator query failed.");
                return false;
            }

            CUDATools::AutoDevice auto_device_exchange(gpuid_);
            auto& tensor = job.mono_tensor->data();
            TRT::CUStream preprocess_stream = nullptr;

            if(tensor == nullptr){
                // not init
                tensor = make_shared<TRT::Tensor>();
                tensor->set_workspace(make_shared<TRT::MixMemory>());

                if(use_multi_preprocess_stream_){
                    checkCudaRuntime(cudaStreamCreate(&preprocess_stream));
                    // owner = true, stream needs to be free during deconstruction
                    tensor->set_stream(preprocess_stream, true);
                }
                else{
                    preprocess_stream = stream_;
                    // owner = false, tensor ignored the stream
                    tensor->set_stream(preprocess_stream, false);
                }
            }

            cv::Size input_size(input_width_, input_height_);
            job.additional.compute(image.size(), input_size);

            preprocess_stream = tensor->get_stream();
            tensor->resize(1, 3, input_height_, input_width_);

            size_t size_image = image.cols * image.rows * 3;
            size_t size_matrix = iLogger::upbound(sizeof(job.additional.d2i), 32);
            auto workspace = tensor->get_workspace();

            uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image);
            float*   affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device         = gpu_workspace + size_matrix;

            uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image);
            float*   affine_matrix_host   = (float*)cpu_workspace;
            uint8_t* image_host           = cpu_workspace + size_matrix;

            memcpy(image_host, image.data, size_image);
            memcpy(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, preprocess_stream));

            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                    image_device, image.cols * 3, image.cols, image.rows,
                    tensor->gpu<float>(), input_width_, input_height_,
                    affine_matrix_device, 114,
                    normalize_, preprocess_stream);
            INFO("666666666666666");
            return true;
        }


    private:
        int gpuid_{0};
        int input_width_{0};
        int input_height_{0};
        float confidence_threshold_{0};
        float nms_threshold_{0};
        int max_objects_{1024};
        NMSMethod nms_method_{NMSMethod::FastGPU};
        TRT::CUStream stream_{nullptr};
        bool use_multi_preprocess_stream_{false};
        CUDAKernel::Norm normalize_;
    };

    std::shared_ptr<Infer> create_infer(const std::string& file, Type type, int gpuid,
                                        float confidence_threshold, float nms_threshole,
                                        NMSMethod nms_method, int max_objects,
                                        bool use_multi_preprocess_stream)
    {
        shared_ptr<InferImpl> instance(new InferImpl{});
        if(!instance->startup(file, type, gpuid, confidence_threshold, nms_threshole,
                              nms_method, max_objects, use_multi_preprocess_stream))
            instance.reset();
        return instance;
    }

}














