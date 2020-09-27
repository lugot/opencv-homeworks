#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iostream>

#include "../include/detector.h"

// Detector class methods
Detector::Detector(cv::Mat input_image){
	this->input_image = input_image;
}

// do nothing in base class, returns a copy of the input image
void Detector::doDetector(){
	result_image = input_image.clone();
}

// get output of the filter
cv::Mat Detector::getResult(){ return result_image; }
cv::Mat Detector::getInput(){ return input_image; }



// Canny Detector methods
EdgesDetector::EdgesDetector(cv::Mat input_image,
		double min_threshold, double max_threshold)
		: Detector(input_image){
	
	setMinThreshold(min_threshold);
	setMaxThreshold(max_threshold);
}

void EdgesDetector::doDetection(){

	auto start = std::chrono::high_resolution_clock::now();
	
	// perform canny
	cv::Canny(input_image, result_image, min_threshold, max_threshold, 3);
	
	auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
			(std::chrono::high_resolution_clock::now() - start);
	
	// infos
	if (DEBUG){
		std::cout << "performed Canny Detector on a " << input_image.size() <<
			" image in " << duration.count()/1000.0 << " msec\n" <<
			"\t thresold1: " << min_threshold << "\n"
			"\t thresold2: " << max_threshold << "\n";
	}
}

// Getters
double EdgesDetector::getMinThreshold(){ return min_threshold; }
double EdgesDetector::getMaxThreshold(){ return max_threshold; }
// Setters
void EdgesDetector::setMinThreshold(double min_threshold){ this->min_threshold = min_threshold; }
void EdgesDetector::setMaxThreshold(double max_threshold){ this->max_threshold = max_threshold; }



// Hough TransformLines methods
LinesDetector::LinesDetector(cv::Mat input_image, double rho, double theta,
		int threshold): Detector(input_image){

	setRho(rho);
	setTheta(theta);
	setThreshold(threshold);
} 

void LinesDetector::doDetection(){

	auto start = std::chrono::high_resolution_clock::now();
	
	// perform hough
	cv::HoughLines(input_image, output_lines, rho, theta, threshold);

	auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
			(std::chrono::high_resolution_clock::now() - start);
	
	// infos
	if (DEBUG){
		std::cout << "performed Hough Transform (Lines) on a " << input_image.size() <<
			" image in " << duration.count()/1000.0 << " msec\n" <<
			"\t     rho: " << rho << " px\n"
			"\t   theta: " << theta*180/CV_PI << " degrees\n" 
			"\tthresold: " << threshold << "\n" <<
            "\t detected " << output_lines.size() << " lines.\n";
	}
}

// Result
std::vector<cv::Vec3f> LinesDetector::getResult(){ return output_lines; }
// Getters
double LinesDetector::getRho(){ return rho; }
double LinesDetector::getTheta(){ return theta; }
int LinesDetector::getThreshold(){ return threshold; }
// Setters
void LinesDetector::setRho(double rho){ this->rho = rho; }
void LinesDetector::setTheta(double theta){ this->theta = theta; }
void LinesDetector::setThreshold(int threshold){ this->threshold = threshold; }



// CirclesDetector methods
CirclesDetector::CirclesDetector(cv::Mat input_image,
	    double dp, double min_dist, double canny_thresh, double acc_thresh)
        : Detector(input_image){
	
	setDp(dp);
	setMinDist(min_dist);
    setCannyThreshold(canny_thresh);
    setAccumulatorThreshold(acc_thresh);
}

void CirclesDetector::doDetection(){

	auto start = std::chrono::high_resolution_clock::now();
	
	// perform hough
	cv::HoughCircles(input_image, output_circles, cv::HOUGH_GRADIENT, 
			dp, min_dist, canny_thresh, acc_thresh);

	auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
			(std::chrono::high_resolution_clock::now() - start);
	
	// infos
	if (DEBUG){
		std::cout << "performed Hough Transform Circles on a " << input_image.size() <<
			" image in " << duration.count()/1000.0 << " msec\n" <<
			"\t          dp: " << dp << "\n"
			"\t    min_dist: " << min_dist << "\n" << 
            "\tcanny_thresh: " << canny_thresh << "\n" << 
            "\t   acc_tresh: " << acc_thresh << "\n" << 
            "detected " << output_circles.size() << " circles\n"; 
	}
}

// Result
std::vector<cv::Vec4f> CirclesDetector::getResult(){ return output_circles; }
// Getters
double CirclesDetector::getDp(){ return dp; }
double CirclesDetector::getMinDist(){ return min_dist; }
double CirclesDetector::getCannyThreshold(){ return canny_thresh; }
double CirclesDetector::getAccumulatorThreshold(){ return acc_thresh; }

    // Setters
void CirclesDetector::setDp(double dp){ this->dp = dp; }
void CirclesDetector::setMinDist(double min_dist){ this->min_dist = min_dist; }
void CirclesDetector::setCannyThreshold(double canny_thresh){ this->canny_thresh = canny_thresh; }
void CirclesDetector::setAccumulatorThreshold(double acc_thresh){ this->acc_thresh = acc_thresh; }
