cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
project(project)

find_package(OpenCV REQUIRED)
#find_package(DGM 1.5 REQUIRED PATHS "$ENV{DGMDIR}/build/install")

include_directories(${OpenCV_INCLUDE_DIRS}
    #${DGM_INCLUDE_DIRS} ***
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

add_library(applib
    src/utils.cpp
    src/tree_localizer.cpp
    src/superpixel.cpp
    src/superpixels_segmenter.cpp
)

add_executable(${PROJECT_NAME}
    src/project.cpp
)

target_link_libraries(${PROJECT_NAME} applib ${OpenCV_LIBS} 
    #${DGM_LIBRARIES} ***
    )
