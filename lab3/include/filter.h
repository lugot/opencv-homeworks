#include <opencv2/core.hpp>

// Generic class implementing a filter with the input and output image data 
//  and the parameters
class Filter{
public:
	Filter(cv::Mat input_image, int filter_size);
	void doFilter();
	cv::Mat getResult();
	void setSize(int filter_size);
	int getSize();

protected:
	cv::Mat input_image, result_image;
	int filter_size;
};

// Derived class for Gaussian filter
class GaussianFilter : public Filter  {
public:
	GaussianFilter(cv::Mat input_image, int, double filter_size);
	void doFilter();
	void setSigma(double sigma);
	double getSigma();

protected:
	double sigma;
};

// Derived class for Median filter
class MedianFilter : public Filter {
public:
	MedianFilter(cv::Mat input_image, int filter_size);
	void doFilter();	
};

// Derived class for Bilateral Filter
class BilateralFilter : public Filter {
public:
	BilateralFilter(cv::Mat input_image, double sigma_range, double sigma_space);
	void doFilter();
	
	void setSigmaRange(double sigma_range);
	void setSigmaSpace(double sigma_space);
	
	double getSigmaRange();
	double getSigmaSpace();
	// void SetSize leaved as a 'backdoor'

protected:
	double sigma_range, sigma_space;
};
