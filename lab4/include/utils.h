// This header contains userful wrapper of imshow
#include <opencv4/opencv2/core.hpp>
#include "global.h"

void imshowHorizontalSided(std::string windowname, cv::Mat img, cv::Mat img2);
void imshowVerticalSided(std::string windowname, cv::Mat img, cv::Mat img2);
void imshowSided(std::string windowname, cv::Mat img, cv::Mat img2);
void showHistogram(std::vector<cv::Mat>& hists); //unimplemented, see imshow_hists
cv::Size adaptedSize(cv::Mat img);

void drawLine(cv::Mat& img, cv::Vec3f line, int size, cv::Scalar color);
void drawCircle(cv::Mat& img, cv::Vec4f circle, int size, cv::Scalar color);

void drawLinesOnImage(cv::Mat& img, std::vector<cv::Vec3f> lines);
void drawCirclesOnImage(cv::Mat& img, std::vector<cv::Vec4f> circles);
