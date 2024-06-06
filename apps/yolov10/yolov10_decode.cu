
#include "trt_common/cuda_tools.hpp"

namespace YOLOV10{

    const int NUM_BOX_ELEMENT = 7;      // left, top, right, bottom, confidence, class, keepflag

    static __device__ void affine_project(const float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }


    /// ------------------ 核函数定义 ------------------
    static __global__ void decode_kernel_yolov10(float* predict, int num_bboxes, float confidence_threshold,
                                                float* invert_affine_matrix, float* parray, int max_objects){

        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= num_bboxes) return;

        float* pitem = predict + 6 * position;
        float confidence = pitem[4];
        int label = pitem[5];

        if(confidence < confidence_threshold)
            return;

        int index = atomicAdd(parray, 1);
        if(index >= max_objects)
            return;

        float left   = *pitem++;
        float top    = *pitem++;
        float right  = *pitem++;
        float bottom = *pitem++;

        affine_project(invert_affine_matrix, left,  top,    &left,  &top);
        affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

        float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = confidence;
        *pout_item++ = label;
        *pout_item++ = 1; // 1 = keep, 0 = ignore
    }


    /// ------------------ 核函数调用 ------------------
    void decode_kernel_invoker(float* predict, int num_bboxes, float confidence_threshold,
                               float* invert_affine_matrix, float* parray,
                               int max_objects, cudaStream_t stream){

        auto grid = CUDATools::grid_dims(num_bboxes);
        auto block = CUDATools::block_dims(num_bboxes);
        checkCudaKernel(decode_kernel_yolov10<<<grid, block, 0, stream>>>(
                predict, num_bboxes, confidence_threshold,
                invert_affine_matrix, parray, max_objects))
    }

}
