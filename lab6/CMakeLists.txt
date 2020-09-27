cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
project(lab6)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

add_library(applib
    src/tracker.cpp
)

add_executable(${PROJECT_NAME}
	src/lab6.cpp
)

target_link_libraries(${PROJECT_NAME} applib ${OpenCV_LIBS})
