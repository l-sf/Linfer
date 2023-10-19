#ifndef LIGHTTRACK_HPP
#define LIGHTTRACK_HPP

#include <iostream>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

namespace LightTrack {

    using namespace std;

    class Tracker{
    public:
        virtual void init(cv::Mat &z_img, cv::Rect &init_bbox) = 0;
        virtual cv::Rect track(cv::Mat& x_img) = 0;
    };

    shared_ptr<Tracker> create_tracker(const std::string &z_path, const std::string &x_path, int gpuid = 0);
};


#endif //LIGHTTRACK_HPP
