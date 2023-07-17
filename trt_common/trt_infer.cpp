

#include "trt_infer.hpp"
#include <cuda_runtime.h>
#include <NvInferRuntime.h>
#include <fstream>
#include <algorithm>
#include <map>
#include "cuda_tools.hpp"
#include "ilogger.hpp"


using namespace std;
using namespace nvinfer1;

class Logger : public ILogger{
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kINTERNAL_ERROR) {
            INFOE("NVInfer INTERNAL_ERROR: %s", msg);
            abort();
        }else if (severity == Severity::kERROR) {
            INFOE("NVInfer: %s", msg);
        }
        else  if (severity == Severity::kWARNING) {
            INFOW("NVInfer: %s", msg);
        }
        else  if (severity == Severity::kINFO) {
            INFOD("NVInfer: %s", msg);
        }
        else {
            INFOD("%s", msg);
        }
    }
};
static Logger gLogger;


namespace TRT {

    template<class T>
    shared_ptr<T> make_nvshared(T* ptr) {
        return shared_ptr<T>(ptr, [](T* p){p->destroy();});
    }

    std::vector<uint8_t> load_file(const string& file) {
        ifstream in(file, ios::in | ios::binary);
        if (!in.is_open())
            return {};
        in.seekg(0, ios::end);
        size_t length = in.tellg();
        std::vector<uint8_t> data;
        if (length > 0){
            in.seekg(0, ios::beg);
            data.resize(length);

            in.read((char*)&data[0], length);
        }
        in.close();
        return data;
    }

    TRT::DataType convert_trt_datatype(nvinfer1::DataType dt){
        switch(dt){
            case nvinfer1::DataType::kFLOAT: return TRT::DataType::Float;
            case nvinfer1::DataType::kHALF: return TRT::DataType::Float16;
            case nvinfer1::DataType::kINT32: return TRT::DataType::Int32;
            default:
                INFOE("Unsupport data type %d", dt);
                return TRT::DataType::Float;
        }
    }

    class EngineContext {
    public:
        EngineContext() = default;
        ~EngineContext() { destroy(); }

        void set_stream(CUStream stream){
            if(owner_stream_){
                if (stream_) cudaStreamDestroy(stream_);
                owner_stream_ = false;
            }
            stream_ = stream;
        }

        bool build_model(const void* pdata, size_t size) {
            destroy();

            if(pdata == nullptr || size == 0) return false;

            owner_stream_ = true;
            checkCudaRuntime(cudaStreamCreate(&stream_));
            if(stream_ == nullptr) return false;

            runtime_ = make_nvshared(createInferRuntime(gLogger));
            if(runtime_ == nullptr) return false;

            engine_ = make_nvshared(runtime_->deserializeCudaEngine(pdata, size, nullptr));
            if(engine_ == nullptr) return false;

            context_ = make_nvshared(engine_->createExecutionContext());
            if(context_ == nullptr) return false;

            return true;
        }

    private:
        void destroy() {
            context_.reset();
            engine_.reset();
            runtime_.reset();
            if(owner_stream_)
                if(stream_) cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }

    public:
        CUStream stream_ = nullptr;
        bool owner_stream_ = false;
        shared_ptr<IExecutionContext> context_ = nullptr;
        shared_ptr<ICudaEngine> engine_ = nullptr;
        shared_ptr<IRuntime> runtime_ = nullptr;
    };

    class InferImpl : public Infer {
    public:
        ~InferImpl();
        bool load(const string& file);
        void destroy();
        void forward(bool sync) override;
        std::shared_ptr<Tensor> input(int index) override;
        std::shared_ptr<Tensor> output(int index) override;
        std::shared_ptr<Tensor> tensor(const std::string& name) override;
        void set_input(int index, std::shared_ptr<Tensor> tensor) override;
        void set_output(int index, std::shared_ptr<Tensor> tensor) override;
        std::string get_input_name(int index) override;
        std::string get_output_name(int index) override;
        bool is_input_name(const std::string& name) override;
        bool is_output_name(const std::string& name) override;
        int num_input() override;
        int num_output() override;
        void print() override;
        void set_stream(CUStream stream) override;
        CUStream get_stream() override;
        int get_max_batch_size() override;
        void synchronize() override;
        int device() override;
        size_t get_device_memory_size() override;
        std::shared_ptr<MixMemory> get_workspace() override;

    private:
        void build_engine_input_and_output_mapper();

    private:
        vector<shared_ptr<Tensor>> inputs_;
        vector<shared_ptr<Tensor>> outputs_;
        vector<string> inputs_names_;
        vector<string> outputs_names_;
        vector<int> inputs_map_to_ordered_index_;
        vector<int> outputs_map_to_ordered_index_;
        vector<shared_ptr<Tensor>> ordered_Blobs_;
        map<string, int> blobs_name_mapper_;
        shared_ptr<EngineContext> context_;
        vector<void*> bindings_ptr_;
        shared_ptr<MixMemory> workspace_;
        int device_id_ = 0;
    };

    /////////////////////////////////////////////////////////////////////////

    InferImpl::~InferImpl() {
        destroy();
    }

    void InferImpl::destroy() {
        int old_device = 0;
        checkCudaRuntime(cudaGetDevice(&old_device));
        checkCudaRuntime(cudaSetDevice(device_id_));
        context_.reset();
        blobs_name_mapper_.clear();
        outputs_.clear();
        inputs_.clear();
        outputs_names_.clear();
        inputs_names_.clear();
        checkCudaRuntime(cudaSetDevice(old_device));
    }

    bool InferImpl::load(const string &file) {
        auto data = load_file(file);
        if(data.empty()) return false;

        context_.reset(new EngineContext{});
        if(!context_->build_model(data.data(), data.size())){
            context_.reset();
            return false;
        }

        workspace_.reset(new MixMemory{});
        cudaGetDevice(&device_id_);

        build_engine_input_and_output_mapper();
        return true;
    }

    void InferImpl::build_engine_input_and_output_mapper() {
        auto* context = (EngineContext*)this->context_.get();
        int nbBindings = context->engine_->getNbBindings();

        inputs_.clear();
        inputs_names_.clear();
        outputs_.clear();
        outputs_names_.clear();
        ordered_Blobs_.clear();
        bindings_ptr_.clear();
        blobs_name_mapper_.clear();

        for(int i = 0; i < nbBindings; ++i){
            auto dims = context->engine_->getBindingDimensions(i);
            auto type = context->engine_->getBindingDataType(i);
            const char* bindingName = context->engine_->getBindingName(i);
            dims.d[0] = 1;
            auto newTensor = make_shared<Tensor>(dims.nbDims, dims.d, convert_trt_datatype(type));
            newTensor->set_stream(context_->stream_);
            newTensor->set_workspace(workspace_);
            if(context->engine_->bindingIsInput(i)){
                // is input
                inputs_.push_back(newTensor);
                inputs_names_.emplace_back(bindingName);
                inputs_map_to_ordered_index_.push_back(ordered_Blobs_.size());
            }
            else{
                // is output
                outputs_.push_back(newTensor);
                outputs_names_.emplace_back(bindingName);
                outputs_map_to_ordered_index_.push_back(ordered_Blobs_.size());
            }
            blobs_name_mapper_[bindingName] = i;
            ordered_Blobs_.push_back(newTensor);
        }

        bindings_ptr_.resize(ordered_Blobs_.size());
    }

    void InferImpl::forward(bool sync) {
        auto* context = (EngineContext*)context_.get();
        int inputBatchSize = inputs_[0]->size(0);

        for(int i = 0; i < context->engine_->getNbBindings(); ++i){
            auto dims = context->engine_->getBindingDimensions(i);
            dims.d[0] = inputBatchSize;
            if(context->engine_->bindingIsInput(i))
                context->context_->setBindingDimensions(i, dims);
        }

        for(auto & output : outputs_){
            output->resize_single_dim(0, inputBatchSize);
            output->to_gpu(false);
        }

        for(int i = 0; i < ordered_Blobs_.size(); ++i)
            bindings_ptr_[i] = ordered_Blobs_[i]->gpu();

        void** bindingsptr = bindings_ptr_.data();
        bool exe_res = context->context_->enqueueV2(bindingsptr, context->stream_, nullptr);
        if(!exe_res){
            auto code = cudaGetLastError();
            INFOF("Enqueue failed, code %d[%s], message %s", code, cudaGetErrorName(code), cudaGetErrorString(code));
        }

        if(sync) synchronize();
    }

    std::shared_ptr<Tensor> InferImpl::input(int index) {
        if(index < 0 || index >= inputs_.size())
            INFOF("Input index[%d] out of range [size=%d]", index, inputs_.size());
        return inputs_[index];
    }

    std::shared_ptr<Tensor> InferImpl::output(int index) {
        if(index < 0 || index >= outputs_.size())
            INFOF("Output index[%d] out of range [size=%d]", index, outputs_.size());
        return outputs_[index];
    }

    std::shared_ptr<Tensor> InferImpl::tensor(const string &name) {
        auto node = blobs_name_mapper_.find(name);
        if(node == blobs_name_mapper_.end())
            INFOF("Could not find the input/output node '%s', please check your model.", name.c_str());
        return ordered_Blobs_[node->second];
    }

    void InferImpl::set_input(int index, std::shared_ptr<Tensor> tensor) {
        if(index < 0 || index >= inputs_.size()){
            INFOF("Input index[%d] out of range [size=%d]", index, inputs_.size());
        }
        inputs_[index] = tensor;
        int order_index = inputs_map_to_ordered_index_[index];
        ordered_Blobs_[order_index] = tensor;
    }

    void InferImpl::set_output(int index, std::shared_ptr<Tensor> tensor) {
        if(index < 0 || index >= outputs_.size()){
            INFOF("Output index[%d] out of range [size=%d]", index, outputs_.size());
        }
        outputs_[index] = tensor;
        int order_index = outputs_map_to_ordered_index_[index];
        ordered_Blobs_[order_index] = tensor;
    }

    std::string InferImpl::get_input_name(int index) {
        if(index < 0 || index >= inputs_names_.size())
            INFOF("Input index[%d] out of range [size=%d]", index, inputs_names_.size());
        return inputs_names_[index];
    }

    std::string InferImpl::get_output_name(int index) {
        if(index < 0 || index >= outputs_names_.size())
            INFOF("Output index[%d] out of range [size=%d]", index, outputs_names_.size());
        return outputs_names_[index];
    }

    bool InferImpl::is_input_name(const string &name) {
        return find(inputs_names_.begin(), inputs_names_.end(), name) != inputs_names_.end();
    }

    bool InferImpl::is_output_name(const string &name) {
        return find(outputs_names_.begin(), outputs_names_.end(), name) != outputs_names_.end();
    }

    int InferImpl::num_input() {
        return inputs_.size();
    }

    int InferImpl::num_output() {
        return outputs_.size();
    }

    void InferImpl::print() {
        if(context_ == nullptr){
            INFOW("Infer print, nullptr...");
            return;
        }

        INFO(" Infer %p detail", this);
        INFO("\tBase device: %s", CUDATools::device_description().c_str());
        INFO("\tMax Batch Size: %d", this->get_max_batch_size());
        INFO("\tInputs: %d", inputs_.size());
        for(int i = 0; i < inputs_.size(); ++i){
            auto& tensor = inputs_[i];
            auto& name = inputs_names_[i];
            INFO("\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
        }
        INFO("\tOutputs: %d", outputs_.size());
        for(int i = 0; i < outputs_.size(); ++i){
            auto& tensor = outputs_[i];
            auto& name = outputs_names_[i];
            INFO("\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
        }
    }

    void InferImpl::set_stream(CUStream stream) {
        context_->set_stream(stream);
        for(auto& t : ordered_Blobs_)
            t->set_stream(stream);
    }

    CUStream InferImpl::get_stream() {
        return context_->stream_;
    }

    int InferImpl::get_max_batch_size() {
        assert(context_ != nullptr);
        return context_->engine_->getMaxBatchSize();
    }

    void InferImpl::synchronize() {
        checkCudaRuntime(cudaStreamSynchronize(context_->stream_));
    }

    int InferImpl::device() {
        return device_id_;
    }

    size_t InferImpl::get_device_memory_size() {
        auto* context = (EngineContext*)this->context_.get();
        return context->context_->getEngine().getDeviceMemorySize();
    }

    std::shared_ptr<MixMemory> InferImpl::get_workspace() {
        return workspace_;
    }

    shared_ptr<Infer> load_infer(const string& file){
        shared_ptr<InferImpl> instance{new InferImpl{}};
        if(!instance->load(file)) instance.reset();
        return instance;
    }

    DeviceMemorySummary get_device_summary() {
        DeviceMemorySummary info{};
        checkCudaRuntime(cudaMemGetInfo(&info.available, &info.total));
        return info;
    }

    int get_device_count() {
        int count = 0;
        checkCudaRuntime(cudaGetDeviceCount(&count));
        return count;
    }

    int get_device() {
        int device = 0;
        checkCudaRuntime(cudaGetDevice(&device));
        return device;
    }

    void set_device(int device_id) {
        if (device_id == -1)
            return;
        checkCudaRuntime(cudaSetDevice(device_id));
    }
}