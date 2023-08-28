
#include "trt_tensor.hpp"
#include <cuda_runtime.h>
#include "cuda_tools.hpp"
#include <cuda_fp16.h>

using namespace std;

namespace TRT{

	inline static int get_device(int device_id){
		if(device_id != CURRENT_DEVICE_ID){
			CUDATools::check_device_id(device_id);
			return device_id;
		}

		checkCudaRuntime(cudaGetDevice(&device_id));
		return device_id;
	}


	///  --------------------------------------------------------------------
	///  ----------------------- MixMemory类的定义 ---------------------------
	///  --------------------------------------------------------------------
	
	MixMemory::MixMemory(int device_id){
		device_id_ = get_device(device_id);
	}

	MixMemory::MixMemory(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size){
		reference_data(cpu, cpu_size, gpu, gpu_size);		
	}

	void MixMemory::reference_data(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size){
		release_all();
		
		if(cpu == nullptr || cpu_size == 0){
			cpu = nullptr;
			cpu_size = 0;
		}

		if(gpu == nullptr || gpu_size == 0){
			gpu = nullptr;
			gpu_size = 0;
		}

		this->cpu_ = cpu;
		this->cpu_size_ = cpu_size;
		this->gpu_ = gpu;
		this->gpu_size_ = gpu_size;

		this->owner_cpu_ = !(cpu && cpu_size > 0);
		this->owner_gpu_ = !(gpu && gpu_size > 0);
		checkCudaRuntime(cudaGetDevice(&device_id_));
	}

	MixMemory::~MixMemory() {
		release_all();
	}

    // 分配 GPU 内存，返回内存地址指针
	void* MixMemory::gpu(size_t size) {

		if (gpu_size_ < size) {
			release_gpu();

			gpu_size_ = size;
			CUDATools::AutoDevice auto_device_exchange(device_id_);
			checkCudaRuntime(cudaMalloc(&gpu_, size));
			checkCudaRuntime(cudaMemset(gpu_, 0, size));
		}
		return gpu_;
	}

    // 分配 CPU 内存，返回内存地址指针
	void* MixMemory::cpu(size_t size) {

		if (cpu_size_ < size) {
			release_cpu();

			cpu_size_ = size;
			CUDATools::AutoDevice auto_device_exchange(device_id_);
			checkCudaRuntime(cudaMallocHost(&cpu_, size));
			Assert(cpu_ != nullptr);
			memset(cpu_, 0, size);
		}
		return cpu_;
	}

	void MixMemory::release_cpu() {
		if (cpu_) {
			if(owner_cpu_){
				CUDATools::AutoDevice auto_device_exchange(device_id_);
				checkCudaRuntime(cudaFreeHost(cpu_));
			}
			cpu_ = nullptr;
		}
		cpu_size_ = 0;
	}

	void MixMemory::release_gpu() {
		if (gpu_) {
			if(owner_gpu_){
				CUDATools::AutoDevice auto_device_exchange(device_id_);
				checkCudaRuntime(cudaFree(gpu_));
			}
			gpu_ = nullptr;
		}
		gpu_size_ = 0;
	}

	void MixMemory::release_all() {
		release_cpu();
		release_gpu();
	}

	const char* data_head_string(DataHead dh){
		switch(dh){
			case DataHead::Init: return "Init";
			case DataHead::Device: return "Device";
			case DataHead::Host: return "Host";
			default: return "Unknow";
		}
	}

	
	///  ---------------------------------------------------------------------
	///  -------------------------- Tensor类的定义 ----------------------------
	///  ---------------------------------------------------------------------
	
	Tensor::Tensor(int n, int c, int h, int w, unique_ptr<MixMemory> data, int device_id) {
		this->device_id_ = get_device(device_id);
		descriptor_string_[0] = 0;
		setup_data(move(data));
		resize(n, c, h, w);
	}

	Tensor::Tensor(const std::vector<int>& dims, unique_ptr<MixMemory> data, int device_id){
		this->device_id_ = get_device(device_id);
		descriptor_string_[0] = 0;
		setup_data(move(data));
		resize(dims);
	}

	Tensor::Tensor(int ndims, const int* dims, unique_ptr<MixMemory> data, int device_id) {
		this->device_id_ = get_device(device_id);
		descriptor_string_[0] = 0;
		setup_data(move(data));
		resize(ndims, dims);
	}

	Tensor::Tensor(unique_ptr<MixMemory> data, int device_id){
		shape_string_[0] = 0;
		descriptor_string_[0] = 0;
		this->device_id_ = get_device(device_id);
		setup_data(move(data));
	}

	Tensor::~Tensor() {
		release();
	}

	const char* Tensor::descriptor() const{
		
		char* descriptor_ptr = (char*)descriptor_string_;
		int device_id = device();
		snprintf(descriptor_ptr, sizeof(descriptor_string_), 
			"Tensor:%p, %s, %s, CUDA:%d", 
			data_.get(),
			shape_string_, 
			device_id
		);
		return descriptor_ptr;
	}

	Tensor& Tensor::compute_shape_string(){

		// clean string
		shape_string_[0] = 0;

		char* buffer = shape_string_;
		size_t buffer_size = sizeof(shape_string_);
		for(int i = 0; i < shape_.size(); ++i){

			int size = 0;
			if(i < shape_.size() - 1)
				size = snprintf(buffer, buffer_size, "%d x ", shape_[i]);
			else
				size = snprintf(buffer, buffer_size, "%d", shape_[i]);

			buffer += size;
			buffer_size -= size;
		}
		return *this;
	}

	void Tensor::reference_data(const vector<int>& shape, void* cpu_data, size_t cpu_size, void* gpu_data, size_t gpu_size){
		data_->reference_data(cpu_data, cpu_size, gpu_data, gpu_size);
		setup_data(move(data_));
		resize(shape);
	}

	void Tensor::setup_data(unique_ptr<MixMemory> data){
		data_ = move(data);
		if(data_ == nullptr){
			data_.reset(new MixMemory{device_id_});
		}else{
			device_id_ = data_->device_id();
		}

		head_ = DataHead::Init;
		if(data_->cpu()){
			head_ = DataHead::Host;
		}

		if(data_->gpu()){
			head_ = DataHead::Device;
		}
	}

	shared_ptr<Tensor> Tensor::clone() const{
		auto new_tensor = make_shared<Tensor>(shape_);
		if(head_ == DataHead::Init)
			return new_tensor;
		
		if(head_ == DataHead::Host){
			memcpy(new_tensor->cpu(), this->cpu(), this->bytes_);
		}else if(head_ == DataHead::Device){
			CUDATools::AutoDevice auto_device_exchange(device());
			checkCudaRuntime(cudaMemcpyAsync(new_tensor->gpu(), this->gpu(), bytes_, cudaMemcpyDeviceToDevice, stream_));
		}
		return new_tensor;
	}

	Tensor& Tensor::copy_from_gpu(size_t offset, const void* src, size_t num_element, int device_id){

		if(head_ == DataHead::Init)
			to_gpu(false);

		size_t offset_location = offset * sizeof(float);
		if(offset_location >= bytes_){
			INFOE("Offset location[%lld] >= bytes_[%lld], out of range", offset_location, bytes_);
			return *this;
		}

		size_t copied_bytes = num_element * sizeof(float);
		size_t remain_bytes = bytes_ - offset_location;
		if(copied_bytes > remain_bytes){
			INFOE("Copied bytes[%lld] > remain bytes[%lld], out of range", copied_bytes, remain_bytes);
			return *this;
		}
		
		if(head_ == DataHead::Device){
			int current_device_id = get_device(device_id);
			int gpu_device_id = device();
			if(current_device_id != gpu_device_id){
				checkCudaRuntime(cudaMemcpyPeerAsync(gpu<unsigned char>() + offset_location, gpu_device_id, src, current_device_id, copied_bytes, stream_));
				//checkCudaRuntime(cudaMemcpyAsync(gpu<unsigned char>() + offset_location, src, copied_bytes, cudaMemcpyDeviceToDevice, stream_));
			}
			else{
				checkCudaRuntime(cudaMemcpyAsync(gpu<unsigned char>() + offset_location, src, copied_bytes, cudaMemcpyDeviceToDevice, stream_));
			}
		}else if(head_ == DataHead::Host){
			CUDATools::AutoDevice auto_device_exchange(this->device());
			checkCudaRuntime(cudaMemcpyAsync(cpu<unsigned char>() + offset_location, src, copied_bytes, cudaMemcpyDeviceToHost, stream_));
		}else{
			INFOE("Unsupported head type %d", head_);
		}
		return *this;
	}

	Tensor& Tensor::copy_from_cpu(size_t offset, const void* src, size_t num_element){

		if(head_ == DataHead::Init)
			to_cpu(false);

		size_t offset_location = offset * sizeof(float);
		if(offset_location >= bytes_){
			INFOE("Offset location[%lld] >= bytes_[%lld], out of range", offset_location, bytes_);
			return *this;
		}

		size_t copied_bytes = num_element * sizeof(float);
		size_t remain_bytes = bytes_ - offset_location;
		if(copied_bytes > remain_bytes){
			INFOE("Copied bytes[%lld] > remain bytes[%lld], out of range", copied_bytes, remain_bytes);
			return *this;
		}

		if(head_ == DataHead::Device){
			CUDATools::AutoDevice auto_device_exchange(this->device());
			checkCudaRuntime(cudaMemcpyAsync((char*)data_->gpu() + offset_location, src, copied_bytes, cudaMemcpyHostToDevice, stream_));
		}else if(head_ == DataHead::Host){
			//checkCudaRuntime(cudaMemcpyAsync((char*)data_->cpu() + offset_location, src, copied_bytes, cudaMemcpyHostToHost, stream_));
			memcpy((char*)data_->cpu() + offset_location, src, copied_bytes);
		}else{
			INFOE("Unsupported head type %d", head_);
		}
		return *this;
	}

	Tensor& Tensor::release() {
		data_->release_all();
		shape_.clear();
		bytes_ = 0;
		head_ = DataHead::Init;
		if(stream_owner_ && stream_ != nullptr){
			CUDATools::AutoDevice auto_device_exchange(this->device());
			checkCudaRuntime(cudaStreamDestroy(stream_));
		}
		stream_owner_ = false;
		stream_ = nullptr;
		return *this;
	}

	bool Tensor::empty() const{
		return data_->cpu() == nullptr && data_->gpu() == nullptr;
	}

	int Tensor::count(int start_axis) const {

		if(start_axis >= 0 && start_axis < shape_.size()){
			int size = 1;
			for (int i = start_axis; i < shape_.size(); ++i) 
				size *= shape_[i];
			return size;
		}else{
			return 0;
		}
	}

	Tensor& Tensor::resize(const std::vector<int>& dims) {
		return resize(dims.size(), dims.data());
	}

	int Tensor::numel() const{
		int value = shape_.empty() ? 0 : 1;
		for(int i : shape_){
			value *= i;
		}
		return value;
	}

	Tensor& Tensor::resize_single_dim(int idim, int size){

		Assert(idim >= 0 && idim < shape_.size());

		auto new_shape = shape_;
		new_shape[idim] = size;
		return resize(new_shape);
	}

	Tensor& Tensor::resize(int ndims, const int* dims) {

		vector<int> setup_dims(ndims);
		for(int i = 0; i < ndims; ++i){
			int dim = dims[i];
			if(dim == -1){
				Assert(ndims == shape_.size());
				dim = shape_[i];
			}
			setup_dims[i] = dim;
		}
		this->shape_ = setup_dims;

		// strides = element_size
		this->strides_.resize(setup_dims.size());
		
		size_t prev_size  = sizeof(float);
		size_t prev_shape = 1;
		for(int i = (int)strides_.size() - 1; i >= 0; --i){
			if(i + 1 < strides_.size()){
				prev_size  = strides_[i+1];
				prev_shape = shape_[i+1];
			}
			strides_[i] = prev_size * prev_shape;
		}

		this->adajust_memory_by_update_dims();
		this->compute_shape_string();
		return *this;
	}

	Tensor& Tensor::adajust_memory_by_update_dims(){
		
		int needed_size = this->numel() * sizeof(float);
		if(needed_size > this->bytes_){
			head_ = DataHead::Init;
		}
		this->bytes_ = needed_size;
		return *this;
	}

	Tensor& Tensor::synchronize(){ 
		CUDATools::AutoDevice auto_device_exchange(this->device());
		checkCudaRuntime(cudaStreamSynchronize(stream_));
		return *this;
	}

	Tensor& Tensor::to_gpu(bool copy) {

		if (head_ == DataHead::Device)
			return *this;

		head_ = DataHead::Device;
		data_->gpu(bytes_);  // 分配 GPU 内存

		if (copy && data_->cpu() != nullptr) {
			CUDATools::AutoDevice auto_device_exchange(this->device());
			checkCudaRuntime(cudaMemcpyAsync(data_->gpu(), data_->cpu(), bytes_, cudaMemcpyHostToDevice, stream_));
		}
		return *this;
	}
	
	Tensor& Tensor::to_cpu(bool copy) {

		if (head_ == DataHead::Host)
			return *this;

		head_ = DataHead::Host;
		data_->cpu(bytes_);

		if (copy && data_->gpu() != nullptr) {
			CUDATools::AutoDevice auto_device_exchange(this->device());
			checkCudaRuntime(cudaMemcpyAsync(data_->cpu(), data_->gpu(), bytes_, cudaMemcpyDeviceToHost, stream_));
			checkCudaRuntime(cudaStreamSynchronize(stream_));
		}
		return *this;
	}

    
	template<typename T>
	static inline void memset_any_type(T* ptr, size_t count, T value){
		for (size_t i = 0; i < count; ++i)
			*ptr++ = value;
	}

	int Tensor::offset_array(size_t size, const int* index_array) const{

		Assert(size <= shape_.size());
		int value = 0;
		for(int i = 0; i < shape_.size(); ++i){

			if(i < size)
				value += index_array[i];

			if(i + 1 < shape_.size())
				value *= shape_[i+1];
		}
		return value;
	}

	int Tensor::offset_array(const std::vector<int>& index_array) const{
		return offset_array(index_array.size(), index_array.data());
	}

	Tensor& Tensor::set_norm_mat(int n, const cv::Mat& image, float mean[3], float std[3]) {
		Assert(image.channels() == 3 && !image.empty());
		Assert(ndims() == 4 && n < shape_[0]);
		to_cpu(false);

		int width   = shape_[3];
		int height  = shape_[2];
		float scale = 1 / 255.0;
		cv::Mat inputframe = image;
		if(inputframe.size() != cv::Size(width, height))
			cv::resize(inputframe, inputframe, cv::Size(width, height));

		if(CV_MAT_DEPTH(inputframe.type()) != CV_32F){
			inputframe.convertTo(inputframe, CV_32F, scale);
		}

		cv::Mat ms[3];
		for (int c = 0; c < 3; ++c)
			ms[c] = cv::Mat(height, width, CV_32F, cpu<float>(n, c));

		split(inputframe, ms);
		Assert((void*)ms[0].data == (void*)cpu<float>(n));

		for (int c = 0; c < 3; ++c)
			ms[c] = (ms[c] - mean[c]) / std[c];
		return *this;
	}

	Tensor& Tensor::set_mat(int n, const cv::Mat& _image) {
		cv::Mat image = _image;
		Assert(!image.empty() && CV_MAT_DEPTH(image.type()) == CV_32F);
		Assert(shape_.size() == 4 && n < shape_[0] && image.channels() == shape_[1]);
		to_cpu(false);

		int width  = shape_[3];
		int height = shape_[2];
		if (image.size() != cv::Size(width, height))
			cv::resize(image, image, cv::Size(width, height));

		if (image.channels() == 1) {
			memcpy(cpu<float>(n), image.data, width * height * sizeof(float));
			return *this;
		}

		vector<cv::Mat> ms(image.channels());
		for (int i = 0; i < ms.size(); ++i) 
			ms[i] = cv::Mat(height, width, CV_32F, cpu<float>(n, i));

		cv::split(image, &ms[0]);
		Assert((void*)ms[0].data == (void*)cpu<float>(n));
		return *this;
	}
    

}; // TRTTensor