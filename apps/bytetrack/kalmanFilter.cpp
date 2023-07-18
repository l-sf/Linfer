#include "kalmanFilter.h"
#include <Eigen/Cholesky>

namespace byte_kalman
{
	const double KalmanFilter::chi2inv95[10] = {
		0,
		3.8415,
		5.9915,
		7.8147,
		9.4877,
		11.070,
		12.592,
		14.067,
		15.507,
		16.919
	};

	Config::Config(){
        
        float std_weight_position_ = 1 / 20.f;
        float std_weight_velocity_ = 1 / 160.f;
        float initiate_state[] = {
            2.0f * std_weight_position_,
            2.0f * std_weight_position_,
            1e-2,
            2.0f * std_weight_position_,
            10.0f * std_weight_velocity_,
            10.0f * std_weight_velocity_,
            1e-5,
            10.0f * std_weight_velocity_,
        };

        float noise[] = {
            std_weight_position_,
            std_weight_position_,
            1e-1,
            std_weight_position_
        };

        float per_frame_motion[] = {
            std_weight_position_,
            std_weight_position_,
            1e-2,
            std_weight_position_,
            std_weight_velocity_,
            std_weight_velocity_,
            1e-5,
            std_weight_velocity_,
        };
        memcpy(this->initiate_state, initiate_state, sizeof(initiate_state));
        memcpy(this->noise, noise, sizeof(noise));
        memcpy(this->per_frame_motion, per_frame_motion, sizeof(per_frame_motion));
    }

    Config& Config::set_initiate_state(const std::vector<float>& values){
        if(values.size() != 8){
            printf("set_initiate_state failed, Values.size(%d0) != 8\n", values.size());
            return *this;
        }
        memcpy(this->initiate_state, values.data(), sizeof(this->initiate_state));
		return *this;
    }

    Config& Config::set_per_frame_motion(const std::vector<float>& values){
        if(values.size() != 8){
            printf("set_per_frame_motion failed, Values.size(%d0) != 8\n", values.size());
            return *this;
        }
        memcpy(this->per_frame_motion, values.data(), sizeof(this->per_frame_motion));
		return *this;
    }

    Config& Config::set_noise(const std::vector<float>& values){
        if(values.size() != 4){
            printf("set_noise failed, Values.size(%d0) != 4\n", values.size());
            return *this;
        }
        memcpy(this->noise, values.data(), sizeof(this->noise));
		return *this;
    }

	KalmanFilter::KalmanFilter()
	{
		int ndim = 4;
		double dt = 1.;

		_motion_mat = Eigen::MatrixXf::Identity(8, 8);
		for (int i = 0; i < ndim; i++) {
			_motion_mat(i, ndim + i) = dt;
		}
		_update_mat = Eigen::MatrixXf::Identity(4, 8);

		this->_std_weight_position = 1. / 20;
		this->_std_weight_velocity = 1. / 160;
	}

	Config& KalmanFilter::config(){
		return this->_config;
	}

	KAL_DATA KalmanFilter::initiate(const DETECTBOX &measurement)
	{
		DETECTBOX mean_pos = measurement;
		DETECTBOX mean_vel;
		for (int i = 0; i < 4; i++) mean_vel(i) = 0;

		KAL_MEAN mean;
		for (int i = 0; i < 8; i++) {
			if (i < 4) mean(i) = mean_pos(i);
			else mean(i) = mean_vel(i - 4);
		}

		KAL_MEAN std;
		std(0) = _config.initiate_state[0] * measurement[3];
		std(1) = _config.initiate_state[1] * measurement[3];
		std(2) = _config.initiate_state[2];
		std(3) = _config.initiate_state[3] * measurement[3];
		std(4) = _config.initiate_state[4] * measurement[3];
		std(5) = _config.initiate_state[5] * measurement[3];
		std(6) = _config.initiate_state[6];
		std(7) = _config.initiate_state[7] * measurement[3];

		KAL_MEAN tmp = std.array().square();
		KAL_COVA var = tmp.asDiagonal();
		return std::make_pair(mean, var);
	}

	void KalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance)
	{
		//revise the data;
		DETECTBOX std_pos;
		std_pos << 
			_config.per_frame_motion[0] * mean(3),
			_config.per_frame_motion[1] * mean(3),
			_config.per_frame_motion[2],
			_config.per_frame_motion[3] * mean(3);

		DETECTBOX std_vel;
		std_vel << 
			_config.per_frame_motion[4] * mean(3),
			_config.per_frame_motion[5] * mean(3),
			_config.per_frame_motion[6],
			_config.per_frame_motion[7] * mean(3);
		KAL_MEAN tmp;
		tmp.block<1, 4>(0, 0) = std_pos;
		tmp.block<1, 4>(0, 4) = std_vel;
		tmp = tmp.array().square();
		KAL_COVA motion_cov = tmp.asDiagonal();
		KAL_MEAN mean1 = this->_motion_mat * mean.transpose();
		KAL_COVA covariance1 = this->_motion_mat * covariance *(_motion_mat.transpose());
		covariance1 += motion_cov;

		mean = mean1;
		covariance = covariance1;
	}

	KAL_HDATA KalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance)
	{
		DETECTBOX std;
		std << _config.noise[0] * mean(3), _config.noise[1] * mean(3),
			_config.noise[2], _config.noise[3] * mean(3);
		KAL_HMEAN mean1 = _update_mat * mean.transpose();
		KAL_HCOVA covariance1 = _update_mat * covariance * (_update_mat.transpose());
		Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
		diag = diag.array().square().matrix();
		covariance1 += diag;
		//    covariance1.diagonal() << diag;
		return std::make_pair(mean1, covariance1);
	}

	KAL_DATA KalmanFilter::update(
			const KAL_MEAN &mean,
			const KAL_COVA &covariance,
			const DETECTBOX &measurement){
		KAL_HDATA pa = project(mean, covariance);
		KAL_HMEAN projected_mean = pa.first;
		KAL_HCOVA projected_cov = pa.second;

		//chol_factor, lower =
		//scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
		//kalmain_gain =
		//scipy.linalg.cho_solve((cho_factor, lower),
		//np.dot(covariance, self._upadte_mat.T).T,
		//check_finite=False).T
		Eigen::Matrix<float, 4, 8> B = (covariance * (_update_mat.transpose())).transpose();
		Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.8x4
		Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean; //eg.1x4
		auto tmp = innovation * (kalman_gain.transpose());
		KAL_MEAN new_mean = (mean.array() + tmp.array()).matrix();
		KAL_COVA new_covariance = covariance - kalman_gain * projected_cov*(kalman_gain.transpose());
		return std::make_pair(new_mean, new_covariance);
	}

	Eigen::Matrix<float, 1, -1> KalmanFilter::gating_distance(
			const KAL_MEAN &mean,
			const KAL_COVA &covariance,
			const std::vector<DETECTBOX> &measurements,
			bool only_position){
		KAL_HDATA pa = this->project(mean, covariance);
		if (only_position) {
			printf("not implement!");
			exit(0);
		}
		KAL_HMEAN mean1 = pa.first;
		KAL_HCOVA covariance1 = pa.second;

		//    Eigen::Matrix<float, -1, 4, Eigen::RowMajor> d(size, 4);
		DETECTBOXSS d(measurements.size(), 4);
		int pos = 0;
		for (DETECTBOX box : measurements) {
			d.row(pos++) = box - mean1;
		}
		Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt().matrixL();
		Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();
		auto zz = ((z.array())*(z.array())).matrix();
		auto square_maha = zz.colwise().sum();
		return square_maha;
	}
}