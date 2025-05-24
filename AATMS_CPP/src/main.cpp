#include "yolo_detector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

void print_usage() {
    std::cout << "Usage: AATMS_CPP.exe <model_path> <video_path> [output_path]" << std::endl;
    std::cout << "  model_path: Path to the YOLO ONNX model file" << std::endl;
    std::cout << "  video_path: Path to the input video file" << std::endl;
    std::cout << "  output_path: (Optional) Path to save the output video" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    std::string model_path = argv[1];
    std::string video_path = argv[2];
    std::string output_path = (argc > 3) ? argv[3] : "";

    try {
        // Initialize YOLO detector
        YoloDetector detector(model_path);

        // Open video capture
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file: " << video_path << std::endl;
            return 1;
        }

        // Get video properties
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);

        // Initialize video writer if output path is provided
        cv::VideoWriter writer;
        if (!output_path.empty()) {
            writer.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                       cv::Size(frame_width, frame_height));
            if (!writer.isOpened()) {
                std::cerr << "Error: Could not create output video file: " << output_path << std::endl;
                return 1;
            }
        }

        // Process video frames
        cv::Mat frame;
        int frame_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (cap.read(frame)) {
            // Detect objects
            auto detections = detector.detect(frame);

            // Draw detections
            detector.draw_detections(frame, detections);

            // Display frame
            cv::imshow("AATMS - Object Detection", frame);

            // Write frame if output path is provided
            if (!output_path.empty()) {
                writer.write(frame);
            }

            // Print progress
            frame_count++;
            if (frame_count % 30 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                    current_time - start_time).count();
                std::cout << "Processed " << frame_count << " frames in " << duration 
                         << " seconds (" << frame_count / duration << " fps)" << std::endl;
            }

            // Check for exit key
            char key = cv::waitKey(1);
            if (key == 27) { // ESC key
                break;
            }
        }

        // Cleanup
        cap.release();
        if (!output_path.empty()) {
            writer.release();
        }
        cv::destroyAllWindows();

        std::cout << "Processing completed. Total frames processed: " << frame_count << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 