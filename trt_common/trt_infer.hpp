

#ifndef TRT_INFER_HPP
#define TRT_INFER_HPP

#include <string>
#include <memory>
#include <vector>
#include <map>
#include "trt_tensor.hpp"

namespace TRT {

    /// ------------------------------------------------------------------------
    /// -------------- 封装接口类，对外只提供接口，具体实现逻辑在子类中 ---------------
    /// ------------------------------------------------------------------------

	class Infer {
	public:
        virtual void forward(bool sync = true) = 0;
        virtual std::shared_ptr<Tensor> input(int index = 0) = 0;
        virtual std::shared_ptr<Tensor> output(int index = 0) = 0;
        virtual std::shared_ptr<Tensor> tensor(const std::string& name) = 0;
        virtual void set_input(int index, std::shared_ptr<Tensor> tensor) = 0;
        virtual void set_output(int index, std::shared_ptr<Tensor> tensor) = 0;
        virtual std::string get_input_name(int index = 0) = 0;
        virtual std::string get_output_name(int index = 0) = 0;
        virtual bool is_input_name(const std::string& name) = 0;
        virtual bool is_output_name(const std::string& name) = 0;
        virtual int num_input() = 0;
        virtual int num_output() = 0;
        virtual void print() = 0;
        virtual void set_stream(CUStream stream) = 0;
        virtual CUStream get_stream() = 0;
        virtual int get_max_batch_size() = 0;
        virtual void synchronize() = 0;
        virtual int device() = 0;
        virtual size_t get_device_memory_size() = 0;
        virtual std::shared_ptr<MixMemory> get_workspace() = 0;
	};

	int get_device_count();
	int get_device();
	void set_device(int device_id);

    /// 加载infer，资源获取即初始化
	std::shared_ptr<Infer> load_infer(const std::string& file);


}


#endif //TRT_INFER_HPP