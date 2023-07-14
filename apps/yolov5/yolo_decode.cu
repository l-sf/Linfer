

#include "trt_common/cuda-tools.hpp"

namespace Yolo {

    const int NUM_BOX_ELEMENT = 7;  // left, top, right, bottom, confidence, class, keepflag


    /// -------------------------------------------------------------------------------------------
    /// -------------------------------------- 核函数定义 ------------------------------------------
    /// -------------------------------------------------------------------------------------------

    static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    static __global__ void decode_kernel(
            float* predict, int num_boxes, int num_classes, float confidence_threshold,
            float* invert_matrix, float* parray, int max_objects){
        int position = blockIdx.x * blockDim.x + threadIdx.x;
        if(position >= num_boxes) return;

        float* pitem = predict + position * (num_classes + 5);
        float objness = pitem[4];
        if(objness < confidence_threshold) return;

        float* pconfidence = pitem + 5;
        float confidence = *pconfidence++;
        int lable = 0;
        for(int i = 1; i < num_classes; ++pconfidence){
            if(*pconfidence > confidence){
                confidence = *pconfidence;
                lable = i;
            }
        }

        confidence *= objness;
        if(confidence < confidence_threshold) return;

        // index记录此线程处理的是第几个box，如果超过最大目标数量，直接退出
        int index = atomicAdd(parray, 1);  // 原子加法，线程安全
        if(index >= max_objects) return;

        float cx = *pitem++;
        float cy = *pitem++;
        float width = *pitem++;
        float height = *pitem++;
        float left = cx - width * 0.5f;
        float top = cy - height * 0.5f;
        float right = cx + width * 0.5f;
        float bottom = cy + height * 0.5f;
        affine_project(invert_matrix, left, top, &left, &top);
        affine_project(invert_matrix, right, bottom, &right, &bottom);

        float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = confidence;
        *pout_item++ = lable;
        *pout_item++ = 1;
    }

    static __device__ float iou_device(float aleft, float atop, float aright, float abottom,
                                float bleft, float btop, float bright, float bbottom){
        float cross_left   = max(aleft, bleft);
        float cross_top    = max(atop, btop);
        float cross_right  = min(aright, bright);
        float cross_bottom = min(abottom, bbottom);

        float cross_area = max(0.0f, cross_right - cross_left) * max(0.0f, cross_bottom - cross_top);
        float union_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop)
                           + max(0.0f, bright - bleft) * max(0.0f, bbottom - btop) - cross_area;
        if(cross_area == 0.f || union_area == 0.f) return 0.0f;
        return cross_area / union_area;
    }

    static __global__ void nms_kernel(float* boxes, int max_objects, float nms_threshold){
        int position = blockIdx.x * blockDim.x + threadIdx.x;
        int count = min((int)*boxes, max_objects);
        if(position >= count) return;

        float* pcurrent = boxes + 1 + position * NUM_BOX_ELEMENT;
        for(int i = 1; i < count; ++i){
            float* pitem = boxes + 1 + i * NUM_BOX_ELEMENT;
            if(i == position || pcurrent[5] != pitem[5]) continue;

            if(pitem[4] >= pcurrent[4]){
                if(pitem[4] == pcurrent[4] && i < position)
                    continue;
                float iou = iou_device(pitem[0], pitem[1], pitem[2], pitem[3],
                                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3]);
                if(iou > nms_threshold){
                    pcurrent[6] = 0;   // 1=keep, 0=ignore
                    return;
                }
            }
        }
    }


    /// ------------------------------------------------------------------------------------------------
    /// ---------------------------------------- 调用核函数 ---------------------------------------------
    /// ------------------------------------------------------------------------------------------------

    void decode_kernel_invoker(float* predict, int num_boxes, int num_classes, float confidence_threshold,
                               float* invert_affine_matrix, float* parray,
                               int max_objects, cudaStream_t stream){
        dim3 grid = CUDATools::grid_dims(num_boxes);
        dim3 block = CUDATools::block_dims(num_boxes);
        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, num_boxes, num_classes, confidence_threshold,
                                                                  invert_affine_matrix, parray, max_objects));
    }

    void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, cudaStream_t stream){
        dim3 grid = CUDATools::grid_dims(max_objects);
        dim3 block = CUDATools::block_dims(max_objects);
        checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, nms_threshold, max_objects));
    }

}
