
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include "trt_common/ilogger.hpp"
#include "trt_common/cuda_tools.hpp"

#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <stack>
#include <memory>
#include <functional>
#include <opencv2/opencv.hpp>

using namespace std;

class LLogger : public nvinfer1::ILogger{
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
static LLogger gLogger;

template<typename _T>
static shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

typedef std::function<void(
        int current, int count, const std::vector<std::string>& files,
        nvinfer1::Dims dims, float* ptensor
)> Int8Process;

// int8熵校准器：用于评估量化前后的分布改变
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2{
public:
    Int8EntropyCalibrator(const vector<string>& imagefiles, nvinfer1::Dims dims, const Int8Process& preprocess) {

        assert(preprocess != nullptr);
        this->dims_ = dims;
        this->allimgs_ = imagefiles;
        this->preprocess_ = preprocess;
        this->fromCalibratorData_ = false;
        files_.resize(dims.d[0]);
    }

    // 这个构造函数，是允许从缓存数据中加载标定结果，这样不用重新读取图像处理
    Int8EntropyCalibrator(const vector<uint8_t>& entropyCalibratorData, nvinfer1::Dims dims, const Int8Process& preprocess) {

        assert(preprocess != nullptr);
        this->dims_ = dims;
        this->entropyCalibratorData_ = entropyCalibratorData;
        this->preprocess_ = preprocess;
        this->fromCalibratorData_ = true;
        files_.resize(dims.d[0]);
    }

    ~Int8EntropyCalibrator() override{
        if(tensor_host_ != nullptr){
            checkCudaRuntime(cudaFreeHost(tensor_host_));
            checkCudaRuntime(cudaFree(tensor_device_));
            tensor_host_ = nullptr;
            tensor_device_ = nullptr;
        }
    }

    // 想要按照多少的batch进行标定
    int getBatchSize() const noexcept override {
        return dims_.d[0];
    }

    bool next() {
        int batch_size = dims_.d[0];
        if (cursor_ + batch_size > allimgs_.size())
            return false;

        for(int i = 0; i < batch_size; ++i)
            files_[i] = allimgs_[cursor_++];

        if(tensor_host_ == nullptr){
            size_t volumn = 1;
            for(int i = 0; i < dims_.nbDims; ++i)
                volumn *= dims_.d[i];

            bytes_ = volumn * sizeof(float);
            checkCudaRuntime(cudaMallocHost(&tensor_host_, bytes_));
            checkCudaRuntime(cudaMalloc(&tensor_device_, bytes_));
        }

        preprocess_(cursor_, allimgs_.size(), files_, dims_, tensor_host_);
        checkCudaRuntime(cudaMemcpy(tensor_device_, tensor_host_, bytes_, cudaMemcpyHostToDevice));
        return true;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        if (!next()) return false;
        bindings[0] = tensor_device_;
        return true;
    }

    const vector<uint8_t>& getEntropyCalibratorData() {
        return entropyCalibratorData_;
    }

    const void* readCalibrationCache(size_t& length) noexcept override {
        if (fromCalibratorData_) {
            length = this->entropyCalibratorData_.size();
            return this->entropyCalibratorData_.data();
        }

        length = 0;
        return nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
        entropyCalibratorData_.assign((uint8_t*)cache, (uint8_t*)cache + length);
    }

private:
    Int8Process preprocess_;
    vector<string> allimgs_;
    size_t batchCudaSize_ = 0;
    int cursor_ = 0;
    size_t bytes_ = 0;
    nvinfer1::Dims dims_{};
    vector<string> files_;
    float* tensor_host_ = nullptr;
    float* tensor_device_ = nullptr;
    vector<uint8_t> entropyCalibratorData_;
    bool fromCalibratorData_ = false;
};

bool build_model(const string& onnx_path, const string& engine_path, const string& img_path){

    if(iLogger::exists(engine_path)){
        printf("Engine has exists.\n");
        return true;
    }

    // 这是基本需要的组件
    auto builder = make_nvshared(nvinfer1::createInferBuilder(gLogger));
    auto config = make_nvshared(builder->createBuilderConfig());

    // createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
    // 因此贯穿以后，请都采用createNetworkV2(1)而非createNetworkV2(0)或者createNetwork
    auto network = make_nvshared(builder->createNetworkV2(1));

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = make_nvshared(nvonnxparser::createParser(*network, gLogger));
    if(!parser->parseFromFile(onnx_path.c_str(), 1)){
        printf("Failed to parse Onnx model.\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }

    int maxBatchSize = 1;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    // 如果模型有多个执行上下文，则必须多个profile
    // 多个输入共用一个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();

    input_dims.d[0] = 1;
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    auto preprocess = [](
            int current, int count, const std::vector<std::string>& files,
            nvinfer1::Dims dims, float* ptensor
    ){
        printf("Preprocess %d / %d\n", count, current);

        // 标定所采用的数据预处理必须与推理时一样
        int width = dims.d[3];
        int height = dims.d[2];
        float mean[] = {0.406, 0.456, 0.485};
        float std[]  = {0.225, 0.224, 0.229};

        for(int i = 0; i < files.size(); ++i){

            auto image = cv::imread(files[i]);
            cv::resize(image, image, cv::Size(width, height));
            int image_area = width * height;
            unsigned char* pimage = image.data;
            float* phost_b = ptensor + image_area * 0;
            float* phost_g = ptensor + image_area * 1;
            float* phost_r = ptensor + image_area * 2;
            for(int i = 0; i < image_area; ++i, pimage += 3){
                // 注意这里的顺序rgb调换了
                *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
                *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
                *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
            }
            ptensor += image_area * 3;
        }
    };

    // 配置int8标定数据读取工具
    vector<string> calFiles = iLogger::find_files(img_path, "*.jpg;*.png;*.bmp;*.jpeg");
    shared_ptr<Int8EntropyCalibrator> calib(new Int8EntropyCalibrator(calFiles, input_dims, preprocess));
    config->setInt8Calibrator(calib.get());

    // 配置最小允许batch
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);

    // 配置最大允许batch
    input_dims.d[0] = maxBatchSize;  // 注意要修改的地方
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    auto model_data = make_nvshared(engine->serialize());
    FILE* f = fopen(engine_path.c_str(), "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    f = fopen("calib.txt", "wb");
    auto calib_data = calib->getEntropyCalibratorData();
    fwrite(calib_data.data(), 1, calib_data.size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    printf("Done.\n");
    return true;
}

bool test_ptq(){
    // 修改这些路径
    string onnx_path = "MixVPR.onnx";
    string engine_path = "MixVPR_int8.engine";
    string img_path = "/home/lsf/Quant_proj/Quant_data/images/val2017";
    if(!build_model(onnx_path, engine_path, img_path)){
        return false;
    }
    return true;
}