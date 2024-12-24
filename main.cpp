#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include "apps/yolo/yolo.hpp"
#include "apps/yolop/yolop.hpp"
#include "apps/rtdetr/rtdetr.hpp" // Include the header for RTDETR

using namespace std;

void performance_v10(const string &engine_file, int gpuid, const string &input_dir);
void batch_inference_v10(const string &engine_file, int gpuid, const string &input_dir, const string &output_dir);
void single_inference_v10(const string &engine_file, int gpuid, const string &input_img, const string &output_img_path);
void performance(const string &engine_file, int gpuid, const string &input_dir);
void batch_inference(const string &engine_file, int gpuid, const string &input_dir, const string &output_dir);
void single_inference(const string &engine_file, int gpuid, const string &input_img, const string &output_img_path);
void performance(const string &engine_file, int gpuid, Yolo::Type type, const string &input_dir);
void batch_inference(const string &engine_file, int gpuid, Yolo::Type type, const string &input_dir, const string &output_dir);
void single_inference(const string &engine_file, int gpuid, Yolo::Type type, const string &input_img, const string &output_img_path);
void inference_bytetrack(const string &engine_file, int gpuid, Yolo::Type type, const string &video_file, const string &output_save_path);
void infer_track(int Mode, const string &path);
void performance_yolop(const string &engine_file, YoloP::Type type, int gpuid, const string &input_dir);
void inference_yolop(const string &engine_file, YoloP::Type type, int gpuid, const string &input_img, const string &output_dir);
void performance_seg(const string &engine_file, int gpuid, const string &input_dir);
void inference_seg(const string &engine_file, int gpuid, const string &input_img, const string &output_img_path);
bool test_ptq();

// Helper function to convert string to Yolo::Type
Yolo::Type stringToYoloType(const string &typeStr)
{
    if (typeStr == "V5")
        return Yolo::Type::V5;
    if (typeStr == "X")
        return Yolo::Type::X;
    if (typeStr == "V7")
        return Yolo::Type::V7;
    if (typeStr == "V8")
        return Yolo::Type::V8;
    throw std::runtime_error("Unknown Yolo type: " + typeStr);
}

// Helper function to convert string to YoloP::Type
YoloP::Type stringToYoloPType(const string &typeStr)
{
    if (typeStr == "V1")
        return YoloP::Type::V1;
    if (typeStr == "V2")
        return YoloP::Type::V2;
    throw std::runtime_error("Unknown YoloP type: " + typeStr);
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <config.yaml>" << endl;
        return 1;
    }

    const string config_file_path = argv[1];

    try
    {
        YAML::Node config = YAML::LoadFile(config_file_path);

        if (!config["tasks"] || !config["tasks"].IsSequence())
        {
            cerr << "Error: Invalid or missing 'tasks' in config.yaml." << endl;
            return 1;
        }

        for (const auto &task_node : config["tasks"])
        {
            string task_name = task_node["task"].as<string>();
            cout << "Executing Task: " << task_name << endl;

            if (!task_node["subtasks"] || !task_node["subtasks"].IsSequence())
            {
                cerr << "Error: Invalid or missing 'subtasks' in config.yaml." << endl;
                continue;
            }

            for (const auto &subtask_node : task_node["subtasks"])
            {
                string subtask_type = subtask_node["type"].as<string>();

                cout << "  Subtask: " << subtask_type << endl;
                if (task_name == "rtdetr")
                {
                    string engine_file = subtask_node["engine_file"].as<string>();
                    int gpuid = subtask_node["gpuid"].as<int>();

                    if (subtask_type == "performance")
                    {
                        string input_dir = subtask_node["input_dir"].as<string>();
                        performance(engine_file, gpuid, input_dir);
                    }
                    else if (subtask_type == "batch_inference")
                    {
                        string input_dir = subtask_node["input_dir"].as<string>();
                        string output_dir = subtask_node["output_dir"].as<string>();
                        batch_inference(engine_file, gpuid, input_dir, output_dir);
                    }
                    else if (subtask_type == "single_inference")
                    {
                        string input_img = subtask_node["input_img"].as<string>();
                        string output_img_path = subtask_node["output_img_path"].as<string>();
                        single_inference(engine_file, gpuid, input_img, output_img_path);
                    }
                    else
                    {
                        cerr << "  Error: Unknown subtask type for rtdetr: " << subtask_type << endl;
                    }
                }
                else if (task_name == "yolov10")
                {
                    string engine_file = subtask_node["engine_file"].as<string>();
                    int gpuid = subtask_node["gpuid"].as<int>();

                    if (subtask_type == "performance_v10")
                    {
                        string input_dir = subtask_node["input_dir"].as<string>();
                        performance_v10(engine_file, gpuid, input_dir);
                    }
                    else if (subtask_type == "batch_inference_v10")
                    {
                        string input_dir = subtask_node["input_dir"].as<string>();
                        string output_dir = subtask_node["output_dir"].as<string>();
                        batch_inference_v10(engine_file, gpuid, input_dir, output_dir);
                    }
                    else if (subtask_type == "single_inference_v10")
                    {
                        string input_img = subtask_node["input_img"].as<string>();
                        string output_img_path = subtask_node["output_img_path"].as<string>();
                        single_inference_v10(engine_file, gpuid, input_img, output_img_path);
                    }
                    else
                    {
                        cerr << "  Error: Unknown subtask type for yolov10: " << subtask_type << endl;
                    }
                }
                else if (task_name == "yolo")
                {
                    string engine_file = subtask_node["engine_file"].as<string>();
                    int gpuid = subtask_node["gpuid"].as<int>();
                    string yolo_type_str = subtask_node["yolo_type"].as<string>();
                    Yolo::Type yolo_type = stringToYoloType(yolo_type_str);

                    if (subtask_type == "batch_inference")
                    {
                        string input_dir = subtask_node["input_dir"].as<string>();
                        string output_dir = subtask_node["output_dir"].as<string>();
                        batch_inference(engine_file, gpuid, yolo_type, input_dir, output_dir);
                    }
                    else if (subtask_type == "performance")
                    {
                        string input_dir = subtask_node["input_dir"].as<string>();
                        performance(engine_file, gpuid, yolo_type, input_dir);
                    }
                    else if (subtask_type == "single_inference")
                    {
                        string input_img = subtask_node["input_img"].as<string>();
                        string output_img_path = subtask_node["output_img_path"].as<string>();
                        single_inference(engine_file, gpuid, yolo_type, input_img, output_img_path);
                    }
                    else
                    {
                        cerr << "  Error: Unknown subtask type for yolo: " << subtask_type << endl;
                    }
                }
                else if (task_name == "track")
                {
                    string engine_file = subtask_node["engine_file"].as<string>();
                    int gpuid = subtask_node["gpuid"].as<int>();
                    string yolo_type_str = subtask_node["yolo_type"].as<string>();
                    Yolo::Type yolo_type = stringToYoloType(yolo_type_str);
                    string video_file = subtask_node["video_file"].as<string>();
                    string output_save_path = subtask_node["output_save_path"].as<string>();

                    if (subtask_type == "inference_bytetrack")
                    {
                        inference_bytetrack(engine_file, gpuid, yolo_type, video_file, output_save_path);
                    }
                    else
                    {
                        cerr << "  Error: Unknown subtask type for track: " << subtask_type << endl;
                    }
                }
                else if (task_name == "yolop")
                {
                    string engine_file = subtask_node["engine_file"].as<string>();
                    int gpuid = subtask_node["gpuid"].as<int>();
                    string yolop_type_str = subtask_node["yolo_type"].as<string>();
                    YoloP::Type yolop_type = stringToYoloPType(yolop_type_str);

                    if (subtask_type == "inference_yolop")
                    {
                        string input_img = subtask_node["input_img"].as<string>();
                        string output_dir = subtask_node["output_dir"].as<string>();
                        inference_yolop(engine_file, yolop_type, gpuid, input_img, output_dir);
                    }
                    else if (subtask_type == "performance_yolop")
                    {
                        string input_dir = subtask_node["input_dir"].as<string>();
                        performance_yolop(engine_file, yolop_type, gpuid, input_dir);
                    }
                    else
                    {
                        cerr << "  Error: Unknown subtask type for yolop: " << subtask_type << endl;
                    }
                }
                else if (task_name == "seg")
                {
                    string engine_file = subtask_node["engine_file"].as<string>();
                    int gpuid = subtask_node["gpuid"].as<int>();
                    if (subtask_type == "inference_seg")
                    {
                        string input_img = subtask_node["input_img"].as<string>();
                        string output_img_path = subtask_node["output_img_path"].as<string>();
                        inference_seg(engine_file, gpuid, input_img, output_img_path);
                    }
                    else if (subtask_type == "performance_seg")
                    {
                        string input_dir = subtask_node["input_dir"].as<string>();
                        performance_seg(engine_file, gpuid, input_dir);
                    }
                    else
                    {
                        cerr << "  Error: Unknown subtask type for seg: " << subtask_type << endl;
                    }
                }
                else
                {
                    cerr << "  Error: Unknown task: " << task_name << endl;
                }
            }
            cout << endl;
        }
    }
    catch (const YAML::Exception &e)
    {
        cerr << "Error parsing YAML file: " << e.what() << endl;
        return 1;
    }
    catch (const std::runtime_error &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}