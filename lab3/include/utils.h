// This header contains userful wrapper of imshow
#include <opencv4/opencv2/core.hpp>

void imshow_hists(std::string windowname, cv::Mat img, std::vector<cv::Mat>& hists);
void imshow_sided(std::string windowname, cv::Mat img, cv::Mat img2);
void showHistogram(std::vector<cv::Mat>& hists); //unimplemented, see imshow_hists
