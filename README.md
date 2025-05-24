# Advanced Automatic Traffic Monitoring System (AATMS)

A comprehensive traffic monitoring system that combines high-performance C++ backend for object detection with a modern .NET frontend. The system uses YOLO for real-time object detection and tracking in traffic videos.

## Project Structure

```
AATMS/
├── AATMS_CPP/              # C++ Backend
│   ├── src/               # Source files
│   │   ├── yolo_detector.hpp
│   │   ├── yolo_detector.cpp
│   │   └── main.cpp
│   ├── CMakeLists.txt     # CMake configuration
│   └── README.md          # C++ backend documentation
├── AATMS.NET/             # .NET Frontend
│   ├── src/              # Source files
│   ├── AATMS.NET.csproj  # Project file
│   └── README.md         # .NET frontend documentation
├── models/               # Model files
│   └── hyp_AdamW_mark003.onnx
├── scripts/             # Utility scripts
│   └── convert_model.py
├── tests/              # Test files
├── .gitignore
├── LICENSE
└── README.md
```

## Features

- Real-time object detection using YOLO
- High-performance C++ backend with OpenCV
- Modern WPF-based .NET frontend
- Support for various YOLO model formats
- Video processing with visualization
- Progress tracking and performance metrics

## Prerequisites

### C++ Backend
- Visual Studio 2022 or later with C++ development tools
- OpenCV 4.8.0 or later
- CMake 3.20 or later
- YOLO ONNX model file

### .NET Frontend
- .NET 6.0 SDK or later
- Visual Studio 2022 or later with .NET development tools

### Python Scripts
- Python 3.9 or later
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AATMS.git
   cd AATMS
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Build the C++ backend:
   ```bash
   cd AATMS_CPP
   mkdir build && cd build
   cmake ..
   cmake --build . --config Release
   ```

4. Build the .NET frontend:
   ```bash
   cd ../AATMS.NET
   dotnet build
   ```

## Usage

### C++ Backend

Run the C++ backend directly:
```bash
AATMS_CPP.exe <model_path> <video_path> [output_path]
```

Example:
```bash
AATMS_CPP.exe models/hyp_AdamW_mark003.onnx videos/traffic.mp4 output.mp4
```

### .NET Frontend

Run the .NET application:
```bash
cd AATMS.NET
dotnet run
```

## Model Conversion

Convert PyTorch models to ONNX format:
```bash
python scripts/convert_model.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenCV](https://opencv.org/)
- [YOLO](https://github.com/ultralytics/yolov5)
- [.NET](https://dotnet.microsoft.com/) 