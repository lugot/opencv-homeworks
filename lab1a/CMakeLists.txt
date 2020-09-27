cmake_minimum_required(VERSION 2.8) 
project(lab1a)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(lab1a
	src/lab1a.cpp)

target_link_libraries(lab1a ${OpenCV_LIBS})
