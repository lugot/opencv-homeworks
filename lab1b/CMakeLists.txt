cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(lab1b)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lab1b
	src/lab1b.cpp)

target_link_libraries(lab1b ${OpenCV_LIBS})
