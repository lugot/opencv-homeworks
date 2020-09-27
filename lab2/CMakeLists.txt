cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(lab2)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} 
	${PROJECT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/include)

add_library(applib
	src/utils.cpp
)
add_executable(lab2
	src/lab2.cpp
)

target_link_libraries(${PROJECT_NAME} applib ${OpenCV_LIBS})
