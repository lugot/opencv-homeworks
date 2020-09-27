#ifndef LAB5__PANORAMIC__UTILS__H
#define LAB5__PANORAMIC__UTILS__H

#include <opencv2/core.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <vector>

class PanoramicUtils {
public:
    static cv::Mat cylindricalProj(const cv::Mat& image, const double angle);
    static void imshowMultiple(cv::String winname, 
            std::vector<cv::Mat> images);
    static void imshowMultipleKeypoints(cv::String winname, 
            std::vector<cv::Mat> images,
            std::vector<std::vector<cv::KeyPoint>> keypoints);
    static void imshowMultipleKeypointsMatched(cv::String winname, 
            std::vector<cv::Mat> images,
            std::vector<std::vector<cv::KeyPoint>> keypoints,
            std::vector<std::vector<cv::DMatch>> mathes);
    static cv::Size adaptedSize(cv::Mat img);
};

#endif // LAB5__PANORAMIC__UTILS__H
