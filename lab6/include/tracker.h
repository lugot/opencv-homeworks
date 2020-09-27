#ifndef tracker_h
#define tracker_h

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

class Tracker{ 
private:
    std::vector<cv::Mat> objects;
    std::vector<std::vector<cv::KeyPoint>> obj_keypoints; 
    std::vector<cv::Mat> obj_descriptors;
    std::vector<std::vector<cv::Point2f>> obj_corners;
    
    cv::Mat prev_frame;
    std::vector<std::vector<cv::Point2f>> objects_pts;
    std::vector<std::vector<cv::Point2f>> points;

    // colors used for the rectangles and for the keypoints
    std::vector<cv::Scalar> colors;

    cv::Mat tracking_mask;
    
    bool draw_keypoints;
    bool draw_tracking;
    double ratio;
    
    void drawRectangle(cv::Mat& img, cv::Mat homography, cv::Size obj_size, cv::Scalar color, int index);
    void drawRectangleFrame(cv::Mat& img,cv::Mat homography,cv::Scalar color, int index);

public:
    // Costructor
    Tracker(std::vector<cv::Mat> objects, double ratio, bool draw_keypoints, bool draw_tracking);
    
    void matchFirstFrame(cv::Mat first_frame, cv::Mat& output_img);
    void trackFrame(cv::Mat next_frame, cv::Mat& output_img);

    static void computeFeatures(cv::Mat img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    static cv::Size adaptedSize(cv::Mat img);
};

#endif // tracker_h
