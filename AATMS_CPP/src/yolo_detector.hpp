#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

class YOLODetector {
public:
    struct Detection {
        cv::Rect bbox;
        float confidence;
        int class_id;
    };

    YOLODetector(const std::string& model_path, float conf_threshold = 0.5f, float nms_threshold = 0.4f);
    ~YOLODetector();

    std::vector<Detection> detect(const cv::Mat& frame);
    void draw_detections(cv::Mat& frame, const std::vector<Detection>& detections);

private:
    cv::dnn::Net net_;
    float conf_threshold_;
    float nms_threshold_;
    std::vector<std::string> class_names_;
    
    void load_model(const std::string& model_path);
    void load_class_names();
    std::vector<Detection> process_output(const cv::Mat& frame, const std::vector<cv::Mat>& outputs);
    void preprocess(const cv::Mat& frame, cv::Mat& blob);
}; 