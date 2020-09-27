cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(lab4)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

add_library(applib
	src/filter.cpp
	src/detector.cpp
	src/callbacks.cpp
	src/utils.cpp
    src/trackbar.cpp
)

add_executable(${PROJECT_NAME}
	src/lab4.cpp
)

target_link_libraries(${PROJECT_NAME} applib ${OpenCV_LIBS})
