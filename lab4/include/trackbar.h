#include <opencv4/opencv2/core.hpp>
#include "../include/global.h"
#include "../include/detector.h"

EdgesDetector edgesDetectorSelector(cv::Mat img);
LinesDetector linesDetectorSelector(cv::Mat img);
CirclesDetector circlesDetectorSelector(cv::Mat img);
