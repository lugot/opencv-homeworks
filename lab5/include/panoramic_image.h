#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/types.hpp>

class PanoramicImage {
public:
    PanoramicImage(double ratio);
    int loadImages(cv::String directory, cv::String file_extension);
    cv::Mat elaborate();
    cv::Mat getResult();
    
    double getRatio();
    void setRatio(double ratio);

private:
    void project();
    int extractFeatures();
    void matchKeypoints();
    int refineMatches();
    void computeTranslations();
    void mergeImages();

    double ratio;
    std::vector<cv::Mat> images;
    double fov;
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<std::pair<cv::Point2f, cv::Point2f>> translations;
    cv::Mat result;
};
