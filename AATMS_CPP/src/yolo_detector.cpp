#include "yolo_detector.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

YOLODetector::YOLODetector(const std::string& model_path, float conf_threshold, float nms_threshold)
    : conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) {
    load_model(model_path);
    load_class_names();
}

YOLODetector::~YOLODetector() {
    net_.release();
}

void YOLODetector::load_model(const std::string& model_path) {
    try {
        net_ = cv::dnn::readNetFromONNX(model_path);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        throw;
    }
}

void YOLODetector::load_class_names() {
    // COCO dataset class names
    class_names_ = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };
}

void YOLODetector::preprocess(const cv::Mat& frame, cv::Mat& blob) {
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(0,0,0), true, false);
}

std::vector<YOLODetector::Detection> YOLODetector::detect(const cv::Mat& frame) {
    cv::Mat blob;
    preprocess(frame, blob);
    
    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    
    return process_output(frame, outputs);
}

std::vector<YOLODetector::Detection> YOLODetector::process_output(
    const cv::Mat& frame, const std::vector<cv::Mat>& outputs) {
    
    std::vector<Detection> detections;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    
    float* data = (float*)outputs[0].data;
    const int rows = outputs[0].size[1];
    const int dimensions = outputs[0].size[2];
    
    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        if (confidence >= conf_threshold_) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, dimensions - 5, CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            
            if (max_class_score > conf_threshold_) {
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                
                int left = static_cast<int>((x - 0.5 * w) * frame.cols);
                int top = static_cast<int>((y - 0.5 * h) * frame.rows);
                int width = static_cast<int>(w * frame.cols);
                int height = static_cast<int>(h * frame.rows);
                
                boxes.push_back(cv::Rect(left, top, width, height));
                scores.push_back(static_cast<float>(max_class_score));
                class_ids.push_back(class_id.x);
            }
        }
        data += dimensions;
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, indices);
    
    for (int idx : indices) {
        Detection det;
        det.bbox = boxes[idx];
        det.confidence = scores[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
    }
    
    return detections;
}

void YOLODetector::draw_detections(cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        cv::rectangle(frame, det.bbox, cv::Scalar(0, 255, 0), 2);
        
        std::string label = class_names_[det.class_id] + ": " + 
                           std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        cv::rectangle(frame, 
                     cv::Point(det.bbox.x, det.bbox.y - label_size.height - 10),
                     cv::Point(det.bbox.x + label_size.width, det.bbox.y),
                     cv::Scalar(0, 255, 0), -1);
        
        cv::putText(frame, label,
                   cv::Point(det.bbox.x, det.bbox.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
} 