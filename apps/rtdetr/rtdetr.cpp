#include "rtdetr.hpp"
#include <queue>
#include <condition_variable>
#include "trt_common/trt_infer.hpp"
#include "trt_common/ilogger.hpp"
#include "trt_common/infer_controller.hpp"
#include "trt_common/preprocess_kernel.cuh"
#include "trt_common/tensor_allocator.hpp"
#include "trt_common/cuda_tools.hpp"


namespace RTDETR{

    using namespace std;

    void decode_kernel_invoker(
            float* predict, int num_bboxes, int num_classes, float confidence_threshold,
            int scale_expand, float* invert_affine_matrix, float* parray,
            int max_objects, cudaStream_t stream
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

    using ControllerImpl = InferController<cv::Mat, BoxArray, tuple<string, int>, AffineMatrix>;

    class InferImpl : public Infer, public ControllerImpl{
    public:

        /** 要求在 InferImpl 里面执行 stop，而不是在基类执行stop **/
        ~InferImpl() override{
            stop();
        }

        bool startup(const string& file, int gpuid, float confidence_threshold,
                     int max_objects, bool use_multi_preprocess_stream){
            normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.f, 0.f, CUDAKernel::ChannelType::Invert);
            confidence_threshold_ = confidence_threshold;
            max_objects_          = max_objects;
            use_multi_preprocess_stream_ = use_multi_preprocess_stream;
            return ControllerImpl::startup(make_tuple(file, gpuid));
        }

        void worker(std::promise<bool>& result) override{
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
            int num_classes = output->shape(2) - 4;
            input_width_  = input->shape(3);
            input_height_ = input->shape(2);
            stream_       = model->get_stream();
            gpu_          = gpuid;
            tensor_allocator_.reset(new TensorAllocator(max_batch_size * 2));

            TRT::Tensor affine_matrix_device{};
            // output_array_device 是输出 output 经过 decode 的结果
            TRT::Tensor output_array_device{};

            /// load success：设置好了输入，宽高，batch，allocator等，返回true
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
                    // mono_tensor 存放预处理之后的 input_image，其附带的 workspace 存放 affine_matrix
                    if(mono_tensor->get_stream() != stream_){
                        // synchronize preprocess stream finish
                        checkCudaRuntime(cudaStreamSynchronize(mono_tensor->get_stream()));
                    }

                    affine_matrix_device.copy_from_gpu(affine_matrix_device.offset(ibatch), mono_tensor->get_workspace()->gpu(), 6);
                    input->copy_from_gpu(input->offset(ibatch), mono_tensor->gpu(), mono_tensor->count());
                    job.mono_tensor->release(); // 释放掉这个mono_tensor
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
                    decode_kernel_invoker(predict_batch, output->shape(1), num_classes, confidence_threshold_,
                                          input_width_, affine_matrix, output_array_ptr, MAX_IMAGE_BBOX, stream_);
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
                tensor->set_workspace(make_shared<TRT::MixMemory>()); // 新创建一个workspace

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
            // 对齐 32 字节
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
        int max_objects_            = 300;
        TRT::CUStream stream_       = nullptr;
        bool use_multi_preprocess_stream_ = false;
        CUDAKernel::Norm normalize_;

    };

    shared_ptr<Infer> create_infer(
            const string& engine_file, int gpuid,
            float confidence_threshold, int max_objects,
            bool use_multi_preprocess_stream
    ){
        shared_ptr<InferImpl> instance(new InferImpl{});
        if(!instance->startup(engine_file, gpuid, confidence_threshold,
                              max_objects, use_multi_preprocess_stream)){
            instance.reset();
        }
        return instance;
    }

} // namespace RTDETR
