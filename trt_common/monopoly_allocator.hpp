/**
 * 分配器
 * 用以实现 tensor内存 复用的问题
 **/

#ifndef MONOPOLY_ALLOCATOR_HPP
#define MONOPOLY_ALLOCATOR_HPP

#include <condition_variable>
#include <vector>
#include <mutex>
#include <memory>

template<class ItemType>
class MonopolyAllocator{
public:

    ///   Data是数据容器类
    ///   允许query获取的item执行item->release释放自身所有权，该对象可以被复用
    ///   通过item->data()获取储存的对象的指针
    class MonopolyData{
    public:
        std::shared_ptr<ItemType>& data(){ return data_; }
        void release(){ manager_->release_one(this); }

    private:
        explicit MonopolyData(MonopolyAllocator* pmanager){ manager_ = pmanager; }

    private:
        friend class MonopolyAllocator;
        MonopolyAllocator* manager_ = nullptr;
        std::shared_ptr<ItemType> data_;
        bool available_ = true;
    };
    using MonopolyDataPointer = std::shared_ptr<MonopolyData>;

    explicit MonopolyAllocator(int size){
        capacity_ = size;
        num_available_ = size;
        datas_.resize(size);

        for(int i = 0; i < size; ++i)
            datas_[i] = std::shared_ptr<MonopolyData>(new MonopolyData(this));
    }

    virtual ~MonopolyAllocator(){
        run_ = false;
        cv_.notify_all();
        
        std::unique_lock<std::mutex> l(lock_);
        cv_exit_.wait(l, [&](){
            return num_wait_thread_ == 0;
        });
    }

    ///   获取一个可用的对象
    ///   timeout：超时时间，如果没有可用的对象，将会进入阻塞等待，如果等待超时则返回空指针
    ///   请求得到一个对象后，该对象被占用，除非他执行了release释放该对象所有权
    MonopolyDataPointer query(int timeout = 10000){

        std::unique_lock<std::mutex> l(lock_);
        if(!run_) return nullptr;
        
        if(num_available_ == 0){
            num_wait_thread_++;

            auto state = cv_.wait_for(l, std::chrono::milliseconds(timeout), [&](){
                return num_available_ > 0 || !run_;
            });

            num_wait_thread_--;
            cv_exit_.notify_one();

            // timeout, no available, exit program
            if(!state || num_available_ == 0 || !run_)
                return nullptr;
        }

        auto item = std::find_if(datas_.begin(), datas_.end(), [](MonopolyDataPointer& item){return item->available_;});
        if(item == datas_.end())
            return nullptr;
        
        (*item)->available_ = false;
        num_available_--;
        return *item;
    }

    int num_available(){
        return num_available_;
    }

    int capacity(){
        return capacity_;
    }

private:
    void release_one(MonopolyData* prq){
        std::unique_lock<std::mutex> l(lock_);
        if(!prq->available_){
            prq->available_ = true;
            num_available_++;
            cv_.notify_one();
        }
    }

private:
    std::mutex lock_;
    std::condition_variable cv_;
    std::condition_variable cv_exit_;
    std::vector<MonopolyDataPointer> datas_;
    int capacity_ = 0;
    volatile int num_available_ = 0;
    volatile int num_wait_thread_ = 0;
    volatile bool run_ = true;
};

#endif // MONOPOLY_ALLOCATOR_HPP