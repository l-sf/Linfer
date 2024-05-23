
#include "preprocess_kernel.cuh"

namespace CUDAKernel{

	Norm Norm::mean_std(const float mean[3], const float std[3], float alpha, ChannelType channel_type){

		Norm out;
		out.type  = NormType::MeanStd;
		out.alpha = alpha;
		out.channel_type = channel_type;
		memcpy(out.mean, mean, sizeof(out.mean));
		memcpy(out.std,  std,  sizeof(out.std));
		return out;
	}

	Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type){

		Norm out;
		out.type = NormType::AlphaBeta;
		out.alpha = alpha;
		out.beta = beta;
		out.channel_type = channel_type;
		return out;
	}

	Norm Norm::None(){
		return {};
	}


    /// ---------------------- 核函数定义 ---------------------

	__global__ void warp_affine_bilinear_and_normalize_plane_kernel(
        uint8_t* src, int src_line_size,
        int src_width, int src_height,
        float* dst, int dst_width, int dst_height,
		uint8_t const_value_st, float* warp_affine_matrix_2_3, Norm norm, int edge){

		int position = blockIdx.x * blockDim.x + threadIdx.x;
		if (position >= edge) return;

		float m_x1 = warp_affine_matrix_2_3[0];
		float m_y1 = warp_affine_matrix_2_3[1];
		float m_z1 = warp_affine_matrix_2_3[2];
		float m_x2 = warp_affine_matrix_2_3[3];
		float m_y2 = warp_affine_matrix_2_3[4];
		float m_z2 = warp_affine_matrix_2_3[5];

		int dx      = position % dst_width;
		int dy      = position / dst_width;
		float src_x = m_x1 * dx + m_y1 * dy + m_z1;
		float src_y = m_x2 * dx + m_y2 * dy + m_z2;
		float c0, c1, c2;

		if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
			// out of range
			c0 = const_value_st;
			c1 = const_value_st;
			c2 = const_value_st;
		}else{
			int y_low = floorf(src_y);
			int x_low = floorf(src_x);
			int y_high = y_low + 1;
			int x_high = x_low + 1;

			uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
			float ly    = src_y - y_low;
			float lx    = src_x - x_low;
			float hy    = 1 - ly;
			float hx    = 1 - lx;
			float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
			uint8_t* v1 = const_value;
			uint8_t* v2 = const_value;
			uint8_t* v3 = const_value;
			uint8_t* v4 = const_value;
			if(y_low >= 0){
				if (x_low >= 0)
					v1 = src + y_low * src_line_size + x_low * 3;

				if (x_high < src_width)
					v2 = src + y_low * src_line_size + x_high * 3;
			}
			
			if(y_high < src_height){
				if (x_low >= 0)
					v3 = src + y_high * src_line_size + x_low * 3;

				if (x_high < src_width)
					v4 = src + y_high * src_line_size + x_high * 3;
			}
			
			// same to opencv
			c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
			c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
			c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
		}

		if(norm.channel_type == ChannelType::Invert){
			float t = c2;
			c2 = c0;  c0 = t;
		}

		if(norm.type == NormType::MeanStd){
			c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
			c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
			c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
		}else if(norm.type == NormType::AlphaBeta){
			c0 = c0 * norm.alpha + norm.beta;
			c1 = c1 * norm.alpha + norm.beta;
			c2 = c2 * norm.alpha + norm.beta;
		}

		int area = dst_width * dst_height;
		float* pdst_c0 = dst + dy * dst_width + dx;
		float* pdst_c1 = pdst_c0 + area;
		float* pdst_c2 = pdst_c1 + area;
		*pdst_c0 = c0;
		*pdst_c1 = c1;
		*pdst_c2 = c2;
	}


    /// ----------------------- 调用核函数 ----------------------------

	void warp_affine_bilinear_and_normalize_plane(
		uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
		float* matrix_2_3, uint8_t const_value, const Norm& norm,
		cudaStream_t stream) {
		
		int jobs   = dst_width * dst_height;
		auto grid  = CUDATools::grid_dims(jobs);
		auto block = CUDATools::block_dims(jobs);
		
		checkCudaKernel(warp_affine_bilinear_and_normalize_plane_kernel <<<grid, block, 0, stream >>> (
			src, src_line_size,
			src_width, src_height, dst,
			dst_width, dst_height, const_value, matrix_2_3, norm, jobs
		));
	}
};
