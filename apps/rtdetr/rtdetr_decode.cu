
#include "trt_common/cuda_tools.hpp"

namespace RTDETR{

    const int NUM_BOX_ELEMENT = 7;      // left, top, right, bottom, confidence, class, keepflag

    static __device__ void affine_project(const float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }


    /// ------------------ 核函数定义 ------------------
    static __global__ void decode_kernel_rtdetr(float* predict, int num_bboxes, int num_classes,
                                                float confidence_threshold, int scale_expand,
                                                float* invert_affine_matrix, float* parray, int max_objects){

        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= num_bboxes) return;

        float* pitem = predict + (4 + num_classes) * position;
        float* class_confidence = pitem + 4;
        float confidence = *class_confidence++;
        int label = 0;
        for(int i = 1; i < num_classes; ++i, ++class_confidence){
            if(*class_confidence > confidence){
                confidence = *class_confidence;
                label = i;
            }
        }

        if(confidence < confidence_threshold)
            return;

        int index = atomicAdd(parray, 1);
        if(index >= max_objects)
            return;

        float cx     = *pitem++;
        float cy     = *pitem++;
        float width  = *pitem++;
        float height = *pitem++;
        float left = (cx - width * 0.5f) * scale_expand;
        float top = (cy - height * 0.5f) * scale_expand;
        float right = (cx + width * 0.5f) * scale_expand;
        float bottom = (cy + height * 0.5f) * scale_expand;

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
    void decode_kernel_invoker(float* predict, int num_bboxes, int num_classes, float confidence_threshold, int scale_expand,
                               float* invert_affine_matrix, float* parray, int max_objects, cudaStream_t stream){

        auto grid = CUDATools::grid_dims(num_bboxes);
        auto block = CUDATools::block_dims(num_bboxes);
        checkCudaKernel(decode_kernel_rtdetr<<<grid, block, 0, stream>>>(
                    predict, num_bboxes, num_classes, confidence_threshold, scale_expand,
                    invert_affine_matrix, parray, max_objects))
    }

}
