cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(lab3)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

add_library(applib
	src/filter.cpp
	src/callbacks.cpp
	src/utils.cpp
)

add_executable(${PROJECT_NAME}
	src/lab3.cpp
)

target_link_libraries(${PROJECT_NAME} applib ${OpenCV_LIBS})
