
#include <iostream>
#include <fstream>
#include <boost/format.hpp>
#include "lighttrack/LightTrack.hpp"
#include "ostrack/OSTrack.hpp"

using namespace std;


inline static cv::Rect read_gt(std::string &gt_file) {
    std::string line;
    cv::Rect gt_box;
    std::ifstream fin(gt_file);  //真实值文件
    getline(fin, line);
    if (!fin)
        std::cout << " Do not read groundtruth!!! " << std::endl;

    std::stringstream line_ss = std::stringstream(line);
    if (line.find(',') != std::string::npos) {
        std::vector<std::string> data;
        std::string tmp;
        while (getline(line_ss, tmp, ','))
            data.push_back(tmp);
        gt_box.x = stoi(data[0]);
        gt_box.y = stoi(data[1]);
        gt_box.width = stoi(data[2]);
        gt_box.height = stoi(data[3]);
    } else
        line_ss >> gt_box.x >> gt_box.y >> gt_box.width >> gt_box.height;

    return gt_box;
}


template<class T>
void LaunchTrack(shared_ptr<T> tracker, int Mode, const string& path){

    cv::VideoCapture cap;
    string display_name = "Track";

    if (Mode == 0 || Mode == 1) {
        cv::Mat frame;
        cap.open(path);
        cap >> frame;
        cv::imshow(display_name, frame);
        cv::putText(frame, "Select target ROI and press ENTER", cv::Point2i(20, 30),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255,0,0), 1);
        cv::Rect init_bbox = cv::selectROI(display_name, frame);
        tracker->init(frame, init_bbox);
        cv::Rect bbox;
        cv::Mat img;
        while (true) {
            bool ret = cap.read(img);
            if (!ret) {
                cout << "----------Read failed!!!----------" << endl;
                return;
            }
            auto start = std::chrono::steady_clock::now();
            bbox = tracker->track(img);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            double time = 1000 * elapsed.count();
            printf("all infer time: %f ms\n", time);
            cv::rectangle(img, bbox, cv::Scalar(0,255,0), 2);
            cv::imshow(display_name, img);
            cv::waitKey(1);
        }
    }
    else if (Mode == 2) {
        string gt_path = "Woman/groundtruth_rect.txt";
        boost::format fmt(path.data());  //数据集图片
        cv::Mat frame;
        frame = cv::imread((fmt % 1).str(),1);
        cv::imshow(display_name, frame);
        cv::Rect init_bbox = read_gt(gt_path);
        tracker->init(frame, init_bbox);
        cv::Rect bbox;
        cv::Mat img;
        for (int i = 1; i < 597; ++i) {
            img = cv::imread((fmt % i).str(),1);
            auto start = std::chrono::steady_clock::now();
            bbox = tracker->track(img);
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            double time = 1000 * elapsed.count();
            printf("all infer time: %f ms\n", time);
            cv::rectangle(img, bbox, cv::Scalar(0,255,0), 2);
            cv::imshow(display_name, img);
            cv::waitKey(1);
        }
        return;
    }
    else {
        printf("Mode错误，0：视频文件；1：摄像头；2：数据集");
        return;
    }
}


int infer_track(int Mode, const string& path){
    string z_path = "lighttrack-z.trt";
    string x_path = "lighttrack-x-head.trt";
    string engine_path = "ostrack-256.trt";

//    auto tracker = LightTrack::create_tracker(z_path, x_path);
    auto tracker = OSTrack::create_tracker(engine_path);
    if(tracker == nullptr){
        printf("tracker is nullptr.\n");
        return -1;
    }

    LaunchTrack(tracker, Mode, path);

    return 0;
}
