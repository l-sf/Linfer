
#ifndef PPSEG_HPP
#define PPSEG_HPP


#include <vector>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include "trt_common/trt_tensor.hpp"


namespace PPSeg {

    using namespace std;

    inline static vector<cv::Scalar_<uchar>> color_map = {
        cv::Scalar_<uchar>(128, 64, 128), // road
        cv::Scalar_<uchar>(244, 35, 232), // sidewalk
        cv::Scalar_<uchar>(70, 70, 70), // building
        cv::Scalar_<uchar>(102, 102, 156), // wall
        cv::Scalar_<uchar>(190, 153, 153), // fence
        cv::Scalar_<uchar>(153, 153, 153), // pole
        cv::Scalar_<uchar>(250, 170, 30), // traffic light
        cv::Scalar_<uchar>(220, 220, 0), // traffic sign
        cv::Scalar_<uchar>(107, 142, 35), // vegetation 植被
        cv::Scalar_<uchar>(152, 251, 152), // terrain
        cv::Scalar_<uchar>(70, 130, 180), // sky
        cv::Scalar_<uchar>(220, 20, 60), // person
        cv::Scalar_<uchar>(255, 0, 0), // rider
        cv::Scalar_<uchar>(0, 0, 142), // car
        cv::Scalar_<uchar>(0, 0, 70), // truck
        cv::Scalar_<uchar>(0, 60, 100), // bus
        cv::Scalar_<uchar>(0, 80, 100), // train
        cv::Scalar_<uchar>(0, 0, 230), // motorcycle
        cv::Scalar_<uchar>(119, 11, 32) // bicycle
    };

    class Seg{
    public:
        virtual cv::Mat seg(const cv::Mat& image) = 0;
    };

    shared_ptr<Seg> create_seg(const string& engine_file, int gpuid = 0);
}


#endif //PPSEG_HPP
