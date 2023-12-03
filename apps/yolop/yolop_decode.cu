
#include "trt_common/cuda_tools.hpp"
#include "yolop.hpp"

namespace YoloP{

    const int NUM_BOX_ELEMENT = 7;   // left, top, right, bottom, confidence, class, keepflag
    
    static __device__ void affine_project(const float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    static __device__ int mini(int a, int b){
        return a < b ? a : b;
    }

    static __device__ int maxi(int a, int b){
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

        float* class_confidence = pitem + 5;
        float confidence        = *class_confidence++;
        int label               = 0;
        for(int i = 1; i < num_classes; ++i, ++class_confidence){
            if(*class_confidence > confidence){
                confidence = *class_confidence;
                label      = i;
            }
        }

        confidence *= objectness;
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
        *pout_item++ = label;
        *pout_item++ = 1; // 1 = keep, 0 = ignore
    }

    static __device__ float iou_device(
            float aleft, float atop, float aright, float abottom,
            float bleft, float btop, float bright, float bbottom
    ){

        float cleft 	= max(aleft, bleft);
        float ctop 		= max(atop, btop);
        float cright 	= min(aright, bright);
        float cbottom 	= min(abottom, bbottom);

        float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
        if(c_area == 0.0f)
            return 0.0f;

        float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
        float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
        return c_area / (a_area + b_area - c_area);
    }

    static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold){

        int position = blockDim.x * blockIdx.x + threadIdx.x;
        int count = min((int)*bboxes, max_objects);
        if (position >= count)
            return;

        // left, top, right, bottom, confidence, class, keepflag
        float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
        for(int i = 0; i < count; ++i){
            float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
            if(i == position || pcurrent[5] != pitem[5]) continue;

            if(pitem[4] >= pcurrent[4]){
                if(pitem[4] == pcurrent[4] && i < position)
                    continue;

                float iou = iou_device(
                        pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                        pitem[0],    pitem[1],    pitem[2],    pitem[3]
                );

                if(iou > threshold){
                    pcurrent[6] = 0;  // 1=keep, 0=ignore
                    return;
                }
            }
        }
    }

    static __global__ void decode_mask_v1_kernel(float* pred_drive, float* pred_lane,
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
        int y = mini(maxi(round(src_y), 0), dst_height);
        int x = mini(maxi(round(src_x), 0), dst_width);

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

    static __global__ void decode_mask_v2_kernel(float* pred_drive, float* pred_lane,
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
        int y = mini(maxi(round(src_y), 0), dst_height);
        int x = mini(maxi(round(src_x), 0), dst_width);

        // 生成mask
        int area = in_width * in_height;
        uint8_t* pdst = pimage_out + dy * dst_width * 3 + dx * 3;
        if(pred_drive[y * in_width + x] < pred_drive[area + y * in_width + x]){
            pdst[0] = 0;
            pdst[1] = 255;
            pdst[2] = 0;
            pdrive_mask_out[dy * dst_width + dx] = 255;
        }
        if(pred_lane[y * in_width + x] > 0.5){
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
                                    int dst_width, int dst_height, Type type, cudaStream_t stream){
        int jobs = dst_width * dst_height;
        auto grid = CUDATools::grid_dims(jobs);
        auto block = CUDATools::block_dims(jobs);
        if(type == Type::V2){
            checkCudaKernel(decode_mask_v2_kernel<<<grid, block, 0, stream>>>(
                    pred_drive, pred_lane, pimage_out, pdrive_mask_out, plane_mask_out,
                    in_width, in_height, affine_matrix, dst_width, dst_height, jobs));
        }
        else{
            checkCudaKernel(decode_mask_v1_kernel<<<grid, block, 0, stream>>>(
                    pred_drive, pred_lane, pimage_out, pdrive_mask_out, plane_mask_out,
                    in_width, in_height, affine_matrix, dst_width, dst_height, jobs));
        }
    }

    void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, cudaStream_t stream){

        auto grid = CUDATools::grid_dims(max_objects);
        auto block = CUDATools::block_dims(max_objects);
        checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(
                parray, max_objects, nms_threshold));
    }
}