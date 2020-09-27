#include "global.h"
#include <opencv2/core.hpp>

// Generic class implementing a filter with the input and output image data 
//  and the parameters
class Detector{
public:
	Detector(cv::Mat input_image);
	void doDetector();
	cv::Mat getInput();
	cv::Mat getResult();

protected:
	cv::Mat input_image, result_image;
};

class EdgesDetector: public Detector {
public:
	EdgesDetector(cv::Mat input_image,
			double min_threshold,
			double max_threshold);
	void doDetection();

	double getMinThreshold();
	double getMaxThreshold();
	int getSobelSize();
	
	void setMinThreshold(double min_threshold);
	void setMaxThreshold(double max_threshold);

private:
	double min_threshold, max_threshold;
};

class LinesDetector: public Detector {
public:
	LinesDetector(cv::Mat input_image,
			double rho,
			double theta,
			int threshold);
	void doDetection();
	std::vector<cv::Vec3f> getResult();

	double getRho();
	double getTheta();
	int getThreshold();
	
	void setRho(double rho);
	void setTheta(double theta);
	void setThreshold(int threshold);

private:
	double rho, theta;
	int threshold;
	std::vector<cv::Vec3f> output_lines;
};

class CirclesDetector: public Detector {
public:
	CirclesDetector(cv::Mat input_image,
			double dp,
			double min_dist,
            double canny_thresh,
            double acc_thresh);
	void doDetection();
	std::vector<cv::Vec4f> getResult();

	double getDp();
	double getMinDist();
    double getCannyThreshold();
    double getAccumulatorThreshold();
	
	void setDp(double dp);
	void setMinDist(double min_dist);
    void setCannyThreshold(double canny_thresh);
    void setAccumulatorThreshold(double acc_thresh);

private:
	double dp, min_dist, canny_thresh, acc_thresh;
	std::vector<cv::Vec4f> output_circles;
};
