cmake_minimum_required(VERSION 3.20)
project(AATMS_CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(AATMS_CPP
    src/main.cpp
    src/yolo_detector.cpp
)

# Link libraries
target_link_libraries(AATMS_CPP ${OpenCV_LIBS})

# Set output directories
set_target_properties(AATMS_CPP PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
)

# Install targets
install(TARGETS AATMS_CPP
    RUNTIME DESTINATION bin
) 