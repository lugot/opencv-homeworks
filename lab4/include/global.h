// This header contains project-wide variables in order to mantain coherence
// among the project
#include <opencv4/opencv2/core.hpp>

// declare global flag
extern int DEBUG;

// screen spatial resolution: (actual screen size - 200px)
extern int SCREEN_HEIGHT;
extern int SCREEN_WIDTH;

// reference image for visualization
extern cv::Mat REFERENCE_IMG;

// window names
#define GENERAL_WINDOW "lab4"
#define EDGE_DETECTOR_WINDOW "canny_window"
#define LINES_DETECTOR_WINDOW "hough_lines_window"
#define CIRCLES_DETECTOR_WINDOW "hough_circles_window"
