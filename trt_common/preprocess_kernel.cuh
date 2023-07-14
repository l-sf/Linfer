

#ifndef YOLO_PREPROCESS_KERNEL_CUH
#define YOLO_PREPROCESS_KERNEL_CUH

#include "cuda-tools.hpp"

namespace CUDAKernel {

    enum class NormType : int {
        None = 0,
        MeanStd = 1,
        AlphaBeta = 2
    };

    enum class ChannelType : int {
        None = 0,
        Invert = 1
    };

    struct Norm {
        float mean[3]{};
        float std[3]{};
        float alpha{}, beta{};
        NormType type = NormType::None;
        ChannelType channel_type = ChannelType::None;

        // out = (x * alpha - mean) / std
        static Norm mean_std(const float mean[3], const float std[3], float alpha = 1/255.f, ChannelType channel_type = ChannelType::None);

        // out = a * alpha + beta
        static Norm alpha_beta(float alpha, float beta = 0.f, ChannelType channel_type = ChannelType::None);

        static Norm None();
    };

    void warp_affine_bilinear_and_normalize_plane(
            uint8_t* src, int src_line_size, int src_width, int src_height,
            float* dst, int dst_width, int dst_height,
            float* matrix_2x3, uint8_t const_value, const Norm& norm,
            cudaStream_t stream);

    void warp_affine_bilinear_and_normalize_focus(
            uint8_t* src, int src_line_size, int src_width, int src_height,
            float* dst, int dst_width, int dst_height,
            float* matrix_2x3, uint8_t const_value, const Norm& norm,
            cudaStream_t stream);

    void resize_bilinear_and_normalize(
            uint8_t* src, int src_line_size, int src_width, int src_height,
            float* dst, int dst_width, int dst_height,
            const Norm& norm, cudaStream_t stream);

    void norm_feature(
            float* feature_array, int num_feature, int feature_length,
            cudaStream_t stream);

    void convert_nv12_to_bgr_invoke(
            const uint8_t* y, const uint8_t* uv, int width, int height,
            int line_size, uint8_t* dst,
            cudaStream_t stream);
}


#endif //YOLO_PREPROCESS_KERNEL_CUH
