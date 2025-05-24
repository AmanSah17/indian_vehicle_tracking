# AATMS C++ Backend

This is the C++ backend implementation of the Advanced Automatic Traffic Monitoring System (AATMS). It provides high-performance object detection and tracking using YOLO and OpenCV.

## Prerequisites

- Visual Studio 2022 or later with C++ development tools
- OpenCV 4.8.0 or later
- CMake 3.20 or later (optional, for building with CMake)
- A YOLO ONNX model file (e.g., YOLOv8n.onnx)

## Building the Project

### Using Visual Studio

1. Open the solution file `AATMS_CPP.sln` in Visual Studio
2. Set the `OPENCV_DIR` environment variable to your OpenCV installation directory
   - For example: `C:\opencv\build`
3. Build the solution in Release mode for x64 platform

### Using CMake

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Usage

The program can be run from the command line with the following arguments:

```bash
AATMS_CPP.exe <model_path> <video_path> [output_path]
```

Arguments:
- `model_path`: Path to the YOLO ONNX model file
- `video_path`: Path to the input video file
- `output_path`: (Optional) Path to save the output video with detections

Example:
```bash
AATMS_CPP.exe models/yolov8n.onnx videos/traffic.mp4 output/result.mp4
```

## Features

- Real-time object detection using YOLO
- Support for various YOLO model formats (ONNX)
- Visualization of detections with bounding boxes and labels
- Progress tracking and performance metrics
- Optional video output with detections

## Integration with .NET Frontend

The C++ backend can be integrated with the .NET frontend using P/Invoke. The following steps are required:

1. Build the C++ project as a DLL
2. Create appropriate C# wrapper classes
3. Use P/Invoke to call the C++ functions from C#

## Performance Considerations

- The program uses OpenCV's DNN module for inference
- CPU backend is used by default for maximum compatibility
- For better performance, consider using CUDA backend if available
- Frame processing speed depends on the model size and hardware capabilities

## Troubleshooting

1. If you get "Could not open video file" error:
   - Check if the video file exists and is not corrupted
   - Ensure you have the necessary codecs installed

2. If you get "Error loading model" error:
   - Verify that the model file exists and is a valid ONNX file
   - Check if the model is compatible with the current OpenCV version

3. If the program runs slowly:
   - Try using a smaller YOLO model (e.g., YOLOv8n instead of YOLOv8x)
   - Consider using GPU acceleration if available
   - Reduce the input video resolution

## License

This project is licensed under the MIT License - see the LICENSE file for details. 