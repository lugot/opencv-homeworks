// This header contains userful wrapper of imshow
#include <opencv2/core.hpp>

void imshowHorizontalSided(std::string windowname, cv::Mat img, cv::Mat img2);
void imshowVerticalSided(std::string windowname, cv::Mat img, cv::Mat img2);
void imshowSided(std::string windowname, cv::Mat img, cv::Mat img2);

cv::Size adaptedSize(cv::Mat img);
double l2dist(cv::Point2f p1, cv::Point2f p2);
void find3DCorners(int nrows, int ncols, std::vector<cv::Point3f> &corners, double square_size);

