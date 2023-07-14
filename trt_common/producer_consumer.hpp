

/// Producer Consumer Model
/// 提供 InferController 类，让具体应用继承
/// 避免重复书写生产者消费者的代码

#ifndef YOLO_PRODUCER_CONSUMER_HPP
#define YOLO_PRODUCER_CONSUMER_HPP

#include <string>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include "trt-infer.hpp"
#include "monopoly_allocator.hpp"


template<class Input, class Output, class StartParam=std::tuple<std::string, int>, class JobAdditional=int>
class InferController {
public:
   struct Job{
       Input input;
       Output output;
       JobAdditional additional;
       MonopolyAllocator<TRT::Tensor>::MonopolyDataPointer mono_tensor;
       std::shared_ptr<std::promise<Output>> pro;
   };

   virtual ~InferController(){
       stop();
   }

   void stop(){
       running_ = false;
       cv_.notify_all();
       {
           std::unique_lock<std::mutex> l(jobs_lock_);
           while (!jobs_.empty()){
               auto& item = jobs_.front();
               if(item.pro)
                   item.pro->set_value(Output{});
               jobs_.pop();
           }
       }

       if(worker_){
           worker_->join();
           worker_.reset();
       }
   }

   bool start_up(const StartParam& param){
       running_ = true;

       std::promise<bool> pro;
       start_param_ = param;
       worker_ = std::make_shared<std::thread>(&InferController::worker, this, std::ref(pro));
       return pro.get_future().get();
   }

   virtual std::shared_future<Output> commit(const Input& input){
       Job job;
       job.pro = std::make_shared<std::promise<Output>>();
       if(!preprocess(job, input)){
           job.pro->set_value(Output{});
           return job.pro->get_future();
       }
       {
           std::unique_lock<std::mutex> l(jobs_lock_);
           jobs_.push(job);
       }
       cv_.notify_one();
       return job.pro->get_future();
   }

   virtual std::vector<std::shared_future<Output>> commits(const std::vector<Input>& inputs){
       int batch_size = std::min((int)inputs.size(), tensor_allocator_->capacity());
       std::vector<Job> jobs(inputs.size());
       std::vector<std::shared_future<Output>> results(inputs.size());

       int numepoch = (inputs.size() + batch_size - 1) / batch_size;
       for(int epoch = 0; epoch < numepoch; ++epoch){
           int begin = epoch * batch_size;
           int end   = std::min((int)inputs.size(), begin + batch_size);

           for(int i = begin; i < end; ++i){
               Job& job = jobs[i];
               job.pro = std::make_shared<std::promise<Output>>();
               if(!preprocess(job, inputs[i])){
                   job.pro->set_value(Output());
               }
               results[i] = job.pro->get_future();
           }

           {
               std::unique_lock<std::mutex> l(jobs_lock_);
               for(int i = begin; i < end; ++i)
                   jobs_.emplace(std::move(jobs[i]));
           }
           cv_.notify_one();
       }
       return results;
   }

protected:
   virtual void worker(std::promise<bool>& pro) = 0;

   virtual bool preprocess(Job& job, const Input& input) = 0;

   virtual bool get_job_and_wait(Job& fetch_job){
       std::unique_lock<std::mutex> l(jobs_lock_);
       cv_.wait(l, [&](){return !running_ || !jobs_.empty();});

       if(!running_) return false;

       fetch_job = std::move(jobs_.front());
       jobs_.pop();
       return true;
   }

   virtual bool get_jobs_and_wait(std::vector<Job>& fetch_jobs, int max_batch_size){
       std::unique_lock<std::mutex> l(jobs_lock_);
       cv_.wait(l, [&](){ return !running_ || !jobs_.empty(); });
       if(!running_) return false;

       fetch_jobs.clear();
       for(int i = 0; i < max_batch_size && !jobs_.empty(); ++i){
           fetch_jobs.emplace_back(std::move(jobs_.front()));
           jobs_.pop();
       }
       return true;
   }

protected:
   StartParam start_param_;
   std::atomic<bool> running_{false};
   std::queue<Job> jobs_;
   std::mutex jobs_lock_;
   std::shared_ptr<std::thread> worker_;
   std::condition_variable cv_;
   std::shared_ptr<MonopolyAllocator<TRT::Tensor>> tensor_allocator_;
};


#endif //YOLO_PRODUCER_CONSUMER_HPP


