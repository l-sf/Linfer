#include "yolo.hpp"
#include <queue>
#include <condition_variable>
#include "trt_common/trt_infer.hpp"
#include "trt_common/ilogger.hpp"
#include "trt_common/infer_controller.hpp"
#include "trt_common/preprocess_kernel.cuh"
#include "trt_common/tensor_allocator.hpp"
#include "trt_common/cuda_tools.hpp"

namespace Yolo{

    using namespace std;

    const char* type_name(Type type){
        switch(type){
            case Type::V5: return "YoloV5";
            case Type::X: return "YoloX";
            case Type::V7: return "YoloV7";
            case Type::V8: return "YoloV8";
            default: return "Unknow";
        }
    }

    void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
        float* invert_affine_matrix, float* parray,
        int max_objects, Type type, cudaStream_t stream
    );

    void nms_kernel_invoker(
        float* parray, float nms_threshold, int max_objects, cudaStream_t stream
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

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
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
            auto& a = boxes[i];
            box_result.emplace_back(a);
            for(int j = i + 1; j < boxes.size(); ++j){
                if(remove_flags[j]) continue;
                auto& b = boxes[j];
                if(b.label == a.label){
                    if(iou(a, b) >= threshold)
                        remove_flags[j] = true;
                }
            }
        }
        return box_result;
    }

    using ControllerImpl = InferController<cv::Mat, BoxArray, tuple<string, int>, AffineMatrix>;

    class InferImpl : public Infer, public ControllerImpl{
    public:

        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        ~InferImpl() override{
            stop();
        }

        virtual bool startup(
            const string& file, Type type, int gpuid, 
            float confidence_threshold, float nms_threshold,
            NMSMethod nms_method, int max_objects,
            bool use_multi_preprocess_stream
        ){
            if(type == Type::V5 || type == Type::V7 || type == Type::V8){
                normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.f, 0.f, CUDAKernel::ChannelType::Invert);
            }
            else if(type == Type::X){
                normalize_ = CUDAKernel::Norm::None();
            }
            else{
                INFOE("Unsupported type %d", type);
            }
            type_ = type;
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;
            nms_method_           = nms_method;
            max_objects_          = max_objects;
            use_multi_preprocess_stream_ = use_multi_preprocess_stream;
            return ControllerImpl::startup(make_tuple(file, gpuid));
        }

        void worker(promise<bool>& result) override{
            // load model
            string file = get<0>(start_param_);
            int gpuid   = get<1>(start_param_);

            TRT::set_device(gpuid);
            auto model = TRT::load_infer(file);
            if(model == nullptr){
                INFOE("Load model failed: %s", file.c_str());
                result.set_value(false);
                return;
            }

            model->print();

            const int MAX_IMAGE_BBOX  = max_objects_;
            const int NUM_BOX_ELEMENT = 7;   // left, top, right, bottom, confidence, class, keepflag
            int max_batch_size = model->get_max_batch_size();
            auto input = model->input();
            auto output = model->output();
            int num_classes;
            if(type_ == Type::V8)   // 84
                num_classes = output->shape(2) - 4;
            else   // 85
                num_classes = output->shape(2) - 5;
            input_width_  = input->shape(3);
            input_height_ = input->shape(2);
            stream_       = model->get_stream();
            gpu_          = gpuid;
            tensor_allocator_.reset(new TensorAllocator(max_batch_size * 2));

            TRT::Tensor affine_matrix_device{};
            // output_array_device 是输出 output 经过 decode 的结果
            TRT::Tensor output_array_device{};

            // load success：设置好了输入，宽高，batch，allocator等，返回true
            result.set_value(true);

            // 先调整 shape 再 分配 input Tensor GPU 内存
            input->resize_single_dim(0, max_batch_size).to_gpu();

            affine_matrix_device.set_stream(stream_);
            // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
            // 先调整 shape 再 分配 affine_matrix Tensor GPU 内存
            affine_matrix_device.resize(max_batch_size, 8).to_gpu();
            // 这里的 1 + MAX_IMAGE_BBOX 结构是，counter + bboxes ...
            // 仍然是先调整shape再分配GPU内存
            output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();

            vector<Job> fetch_jobs;
            while(get_jobs_and_wait(fetch_jobs, max_batch_size)){
                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);

                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job  = fetch_jobs[ibatch];
                    auto& mono_tensor = job.mono_tensor->data();

                    if(mono_tensor->get_stream() != stream_){
                        // synchronize preprocess stream finish
                        checkCudaRuntime(cudaStreamSynchronize(mono_tensor->get_stream()));
                    }

                    affine_matrix_device.copy_from_gpu(affine_matrix_device.offset(ibatch), mono_tensor->get_workspace()->gpu(), 6);
                    input->copy_from_gpu(input->offset(ibatch), mono_tensor->gpu(), mono_tensor->count());
                    job.mono_tensor->release();
                }
                
                // 进行推理，一次推理一批
                model->forward(false);
                
                output_array_device.to_gpu(false);
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    
                    auto& job= fetch_jobs[ibatch];
                    float* predict_batch = output->gpu<float>(ibatch);
                    float* output_array_ptr   = output_array_device.gpu<float>(ibatch);
                    auto affine_matrix = affine_matrix_device.gpu<float>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
                    decode_kernel_invoker(predict_batch, output->size(1), num_classes, confidence_threshold_,
                                          affine_matrix, output_array_ptr, MAX_IMAGE_BBOX, type_, stream_);

                    if(nms_method_ == NMSMethod::CUDA){
                        nms_kernel_invoker(output_array_ptr, nms_threshold_, MAX_IMAGE_BBOX, stream_);
                    }
                }

                output_array_device.to_cpu();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    float* parray = output_array_device.cpu<float>(ibatch);
                    int count     = min(MAX_IMAGE_BBOX, (int)*parray);
                    auto& job = fetch_jobs[ibatch];
                    auto& image_based_boxes = job.output;
                    for(int i = 0; i < count; ++i){
                        float* pbox  = parray + 1 + i * NUM_BOX_ELEMENT;
                        int label    = pbox[5];
                        int keepflag = pbox[6];
                        if(keepflag == 1){
                            image_based_boxes.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                        }
                    }

                    if(nms_method_ == NMSMethod::CPU){
                        image_based_boxes = cpu_nms(image_based_boxes, nms_threshold_);
                    }
                    job.pro->set_value(image_based_boxes);
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFO("Engine destroy.");
        }

        bool preprocess(Job& job, const cv::Mat& image) override{
            if(tensor_allocator_ == nullptr){
                INFOE("tensor_allocator_ is nullptr.");
                return false;
            }
            if(image.empty()){
                INFOE("Image is empty.");
                return false;
            }
            // 向 allocator 申请一个 tensor
            job.mono_tensor = tensor_allocator_->query();
            if(job.mono_tensor == nullptr){
                INFOE("Tensor allocator query failed.");
                return false;
            }

            CUDATools::AutoDevice auto_device(gpu_);
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
                }else{
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
            uint8_t* image_device         = size_matrix + gpu_workspace;

            uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image);
            float* affine_matrix_host     = (float*)cpu_workspace;
            uint8_t* image_host           = size_matrix + cpu_workspace;

            //checkCudaRuntime(cudaMemcpyAsync(image_host,   image.data, size_image, cudaMemcpyHostToHost,   stream_));
            // speed up
            memcpy(image_host, image.data, size_image);
            memcpy(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, preprocess_stream));

            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                image_device,         image.cols * 3,       image.cols,       image.rows, 
                tensor->gpu<float>(), input_width_,         input_height_, 
                affine_matrix_device, 114, 
                normalize_, preprocess_stream
            );
            return true;
        }

        vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) override{
            return ControllerImpl::commits(images);
        }

        std::shared_future<BoxArray> commit(const cv::Mat& image) override{
            return ControllerImpl::commit(image);
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_        = 0;
        int max_objects_            = 1024;
        NMSMethod nms_method_       = NMSMethod::CUDA;
        TRT::CUStream stream_       = nullptr;
        bool use_multi_preprocess_stream_ = false;
        CUDAKernel::Norm normalize_;
        Type type_;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, Type type, int gpuid, 
        float confidence_threshold, float nms_threshold,
        NMSMethod nms_method, int max_objects,
        bool use_multi_preprocess_stream
    ){
        shared_ptr<InferImpl> instance(new InferImpl{});
        if(!instance->startup(engine_file, type, gpuid, confidence_threshold,
                              nms_threshold, nms_method, max_objects, use_multi_preprocess_stream)){
            instance.reset();
        }
        return instance;
    }

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, Type type, int ibatch){

        CUDAKernel::Norm normalize;
        if(type == Type::V5){
            normalize = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
        }else if(type == Type::X){
            //float mean[] = {0.485, 0.456, 0.406};
            //float std[]  = {0.229, 0.224, 0.225};
            //normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);
            normalize = CUDAKernel::Norm::None();
        }else{
            INFOE("Unsupport type %d", type);
        }
        
        cv::Size input_size(tensor->size(3), tensor->size(2));
        AffineMatrix affine;
        affine.compute(image.size(), input_size);

        size_t size_image      = image.cols * image.rows * 3;
        size_t size_matrix     = iLogger::upbound(sizeof(affine.d2i), 32);
        auto workspace         = tensor->get_workspace();
        uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image);
        float*   affine_matrix_device = (float*)gpu_workspace;
        uint8_t* image_device         = size_matrix + gpu_workspace;

        uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image);
        float* affine_matrix_host     = (float*)cpu_workspace;
        uint8_t* image_host           = size_matrix + cpu_workspace;
        auto stream                   = tensor->get_stream();

        memcpy(image_host, image.data, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));
        checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream));

        CUDAKernel::warp_affine_bilinear_and_normalize_plane(
            image_device,               image.cols * 3,       image.cols,       image.rows, 
            tensor->gpu<float>(ibatch), input_size.width,     input_size.height, 
            affine_matrix_device, 114, 
            normalize, stream
        );
        tensor->synchronize();
    }
};