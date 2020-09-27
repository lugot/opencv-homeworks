#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iostream>
#include "../include/global.h"
#include "../include/filter.h"

// Filter class methods
Filter::Filter(cv::Mat input_image, int filter_size) {
	this->input_image = input_image;
	this->setSize(filter_size);	
}

// do nothing in base class, returns a copy of the input image
void Filter::doFilter() {
	result_image = input_image.clone();
}

// get output of the filter
cv::Mat Filter::getResult() {
	return result_image;
}

//set window size (it needs to be odd)
void Filter::setSize(int filter_size) {
	if (filter_size % 2 == 0)
		filter_size++;
	this->filter_size = filter_size;
}

//get window size 
int Filter::getSize() {
	return filter_size;
}

// Gaussian Filter class methods
GaussianFilter::GaussianFilter(cv::Mat input_image, int filter_size,
		double sigma): Filter(input_image, filter_size){
	
	setSigma(sigma);
}

void GaussianFilter::doFilter(){

	auto start = std::chrono::high_resolution_clock::now();
	
	// no filter if sigma == 0.0
	if (sigma == 0.0) result_image = input_image.clone();
	// perform filter
	else cv::GaussianBlur(input_image, result_image, 
			cv::Size(filter_size, filter_size), sigma);

	auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
			(std::chrono::high_resolution_clock::now() - start);
	
	// infos
	if (DEBUG){
		std::cout << "performed Gaussian Filter on a " << input_image.size() <<
			" image in " << duration.count()/1000.0 << " msec\n" <<
			"\tfilter size: " << filter_size << "\n"
			"\t      sigma: " << sigma << "\n" << std::flush;
	}
}

void GaussianFilter::setSigma(double sigma){
	this->sigma = sigma;
}
double GaussianFilter::getSigma(){ return sigma; }

// Median Filter class methods
MedianFilter::MedianFilter(cv::Mat input_img, int size)
		:Filter(input_img, size){
	// do nothing 
}
	
void MedianFilter::doFilter(){
	
	auto start = std::chrono::high_resolution_clock::now();
	
	// perform filter
	cv::medianBlur(input_image, result_image, filter_size);
	
	auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
			(std::chrono::high_resolution_clock::now() - start);
	
	// infos
	if (DEBUG){
		std::cout << "performed Median Filter on a " << input_image.size() <<
			" image in " << duration.count()/1000.0 << " msec\n" <<
			"\tfilter size: " << filter_size << "\n" << std::flush;
	}
}


// Bilateral Filter class methods
BilateralFilter::BilateralFilter(cv::Mat input_image, 
		double sigma_range, double sigma_space)
		: Filter(input_image, 6*static_cast<int>(sigma_space)){
	
	setSigmaRange(sigma_range);
	setSigmaSpace(sigma_space);
}

void BilateralFilter::doFilter(){
	
	auto start = std::chrono::high_resolution_clock::now();
	
	// perform filter
	cv::bilateralFilter(input_image, result_image, filter_size, 
			sigma_range, sigma_space);
	
	auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
			(std::chrono::high_resolution_clock::now() - start);
	
	// infos
	if (DEBUG){
		std::cout << "performed Bilateral Filter on a " << input_image.size() <<
			" image in " << duration.count()/1000.0 << " msec\n" <<
			"\tfilter size: " << filter_size << "\n"
			"\tsigma range: " << sigma_range << "\n" 
			"\tsigma space: " << sigma_space << "\n" << std::flush;
	}
}

void BilateralFilter::setSigmaRange(double sigma_range){
	this->sigma_range = sigma_range;
}
void BilateralFilter::setSigmaSpace(double sigma_space){
	this->sigma_space = sigma_space;
	setSize(6*static_cast<int>(sigma_space)); // as requested
}

double BilateralFilter::getSigmaRange(){ return sigma_range; }
double BilateralFilter::getSigmaSpace(){ return sigma_space; }


/* garbage implementation of median filtering, too slow
 *
 *
 *
 *    int window[filter_size][filter_size];
 *    result_image = input_image.clone();
 *    int radius = filter_size/2;
 *
 *    std::cout << "size " << input_image.rows << " " << input_image.cols << std::endl;
 *
 *    for(int channel=0; channel<3; channel++){
 *        
 *        for(int y=radius; y<input_image.cols-radius; y++){
 *            
 *
 *            for(int x=radius; x<input_image.rows-radius; x++){
 *                std::vector<int> px_values;
 *        
 *                for(int i=x-filter_size/2; i<=x+filter_size/2; i++){
 *                    for(int j=y-filter_size/2; j<=y+filter_size/2; j++ ){
 *                        
 *                        if (i<0 or j<0 or i>=input_image.rows or j>=input_image.cols) continue;
 *
 *                        px_values.push_back(input_image.at<cv::Vec3b>(i,j)[channel]);
 *                    }
 *                }
 *
 *                std::sort(px_values.begin(), px_values.end());
 *
 *                int median = px_values[px_values.size()/2];
 *                result_image.at<cv::Vec3b>(x,y)[channel] = median;
 *            
 *            }} //window double for
 *    } //channel for
 */
