#pragma once

#include "dataType.h"

namespace byte_kalman
{
	struct Config{
		// kalman
		// /** 初始状态 **/
		float initiate_state[8];

		// /** 每一侦的运动量协方差，下一侦 = 当前帧 + 运动量 **/
		float per_frame_motion[8];

		// /** 测量噪声，把输入映射到测量空间中后的噪声 **/
		float noise[4];

		float track_thresh = 0.5;
		float high_thresh = 0.6;
		float match_thresh = 0.8;
		int max_time_lost = 30;

		Config& set_initiate_state(const std::vector<float>& values);
		Config& set_per_frame_motion(const std::vector<float>& values);
		Config& set_noise(const std::vector<float>& values);
		Config& set_track_thresh(float value){this->track_thresh = value; return *this;};
		Config& set_high_thresh(float value){this->high_thresh = value; return *this;};
		Config& set_match_thresh(float value){this->match_thresh = value; return *this;};
		Config& set_max_time_lost(int value){this->max_time_lost = value; return *this;};

		Config();
	};

	class KalmanFilter
	{
	public:
		static const double chi2inv95[10];
		KalmanFilter();

		Config& config();
		KAL_DATA initiate(const DETECTBOX& measurement);
		void predict(KAL_MEAN& mean, KAL_COVA& covariance);
		KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance);
		KAL_DATA update(const KAL_MEAN& mean,
			const KAL_COVA& covariance,
			const DETECTBOX& measurement);

		Eigen::Matrix<float, 1, -1> gating_distance(
			const KAL_MEAN& mean,
			const KAL_COVA& covariance,
			const std::vector<DETECTBOX>& measurements,
			bool only_position = false);

	private:
		Config _config;
		Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
		Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
		float _std_weight_position;
		float _std_weight_velocity;
	};
}