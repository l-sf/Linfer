
#include "trt_common/cuda_tools.hpp"
#include "yolop.hpp"

namespace YoloP{

    const int NUM_BOX_ELEMENT = 6;    // left, top, right, bottom, confidence, keepflag
    static __device__ void affine_project(const float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }


    /// ------------------ 核函数定义 ------------------

    static __global__ void decode_box_kernel(float* predict, int num_bboxes, int num_classes, float confidence_threshold,
                                                float* invert_affine_matrix, float* parray, int max_objects){

        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= num_bboxes) return;

        float* pitem = predict + (5 + num_classes) * position;
        float objectness = pitem[4];
        if(objectness < confidence_threshold)
            return;

        float* ptr_class = pitem + 5;
        float confidence = *ptr_class * objectness;
        if(confidence < confidence_threshold)
            return;

        int index = atomicAdd(parray, 1);
        if(index >= max_objects)
            return;

        float cx     = *pitem++;
        float cy     = *pitem++;
        float width  = *pitem++;
        float height = *pitem++;
        float left   = cx - width * 0.5f;
        float top    = cy - height * 0.5f;
        float right  = cx + width * 0.5f;
        float bottom = cy + height * 0.5f;
        affine_project(invert_affine_matrix, left,  top,    &left,  &top);
        affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

        float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = confidence;
        *pout_item++ = 1; // 1 = keep, 0 = ignore
    }

    static __global__ void decode_drive_lane_kernel(float* pred_drive, float* parray_drive,
                                                    float* pred_lane, float* parray_lane, int area){

        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= area) return;

        float* pitem = pred_drive + position;
        float* pout_item = parray_drive + position;
        *pout_item = *pitem < *(pitem + area) ? 1 : 0;

        pitem = pred_lane + position;
        pout_item = parray_lane + position;
        *pout_item = *pitem < *(pitem + area) ? 1 : 0;
    }


    /// ------------------ 核函数调用 ------------------

    void decode_box_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold,
                                   float* invert_affine_matrix, float* parray,
                                   int max_objects, cudaStream_t stream){

        auto grid = CUDATools::grid_dims(num_bboxes);
        auto block = CUDATools::block_dims(num_bboxes);
        checkCudaKernel(decode_box_kernel<<<grid, block, 0, stream>>>(
                    predict, num_bboxes, num_classes, confidence_threshold, invert_affine_matrix, parray, max_objects));

    }

    void decode_drive_lane_kernel_invoker(float* pred_drive, float* parray_drive,
                                          float* pred_lane, float* parray_lane,
                                          int height, int width, cudaStream_t stream){
        int area = height * width;
        auto grid = CUDATools::grid_dims(area);
        auto block = CUDATools::block_dims(area);
        checkCudaKernel(decode_drive_lane_kernel<<<grid, block, 0, stream>>>(
                            pred_drive, parray_drive, pred_lane, parray_lane, area));
    }
};