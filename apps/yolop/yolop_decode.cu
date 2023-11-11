
#include "trt_common/cuda_tools.hpp"
#include "trt_common/ilogger.hpp"
#include "yolop.hpp"

namespace YoloP{

    const int NUM_BOX_ELEMENT = 5;    // left, top, right, bottom, confidence
    
    static __device__ void affine_project(const float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    static __device__ int min(int a, int b){
        return a < b ? a : b;
    }

    static __device__ int max(int a, int b){
        return a < b ? b : a;
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
    }

    static __global__ void decode_mask_kernel(float* pred_drive, float* pred_lane,
                                              uint8_t* pimage_out, uint8_t* pdrive_mask_out, uint8_t* plane_mask_out,
                                              int in_width, int in_height, float* affine_matrix,
                                              int dst_width, int dst_height, int edge){

        int position = blockDim.x * blockIdx.x + threadIdx.x;
        if (position >= edge) return;

        int dx = position % dst_width;
        int dy = position / dst_width;

        // 映射
        float src_x, src_y;
        affine_project(affine_matrix, dx, dy, &src_x, &src_y);
        // 边界判断
        int y = min(max(round(src_y), 0), dst_height);
        int x = min(max(round(src_x), 0), dst_width);

        // 生成mask
        int area = in_width * in_height;
        uint8_t* pdst = pimage_out + dy * dst_width * 3 + dx * 3;
        if(pred_drive[y * in_width + x] < pred_drive[area + y * in_width + x]){
            pdst[0] = 0;
            pdst[1] = 255;
            pdst[2] = 0;
            pdrive_mask_out[dy * dst_width + dx] = 255;
        }
        if(pred_lane[y * in_width + x] < pred_lane[area + y * in_width + x]){
            pdst[0] = 255;
            pdst[1] = 0;
            pdst[2] = 0;
            plane_mask_out[dy * dst_width + dx] = 255;
        }
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

    void decode_mask_kernel_invoker(float* pred_drive, float* pred_lane,
                                    uint8_t* pimage_out, uint8_t* pdrive_mask_out, uint8_t* plane_mask_out,
                                    int in_width, int in_height, float* affine_matrix,
                                    int dst_width, int dst_height, cudaStream_t stream){
        int jobs = dst_width * dst_height;
        auto grid = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);
        checkCudaKernel(decode_mask_kernel<<<grid, block, 0, stream>>>(
                            pred_drive, pred_lane, pimage_out, pdrive_mask_out, plane_mask_out,
                            in_width, in_height, affine_matrix, dst_width, dst_height, jobs));
    }
}