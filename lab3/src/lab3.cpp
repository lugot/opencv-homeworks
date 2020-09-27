#include <bits/stdc++.h>
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/matx.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/core/utility.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/utils/filesystem.hpp>
#include "../include/global.h"
#include "../include/filter.h"
#include "../include/utils.h"
#include "../include/callbacks.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv){

	// --- PART1: HISTOGRAM EQUALIZATION ---
	
	// Loading image
	//string filename = "../data/lena.png";
	//string filename = "../data/countryside.jpg";
	//string filename = "../data/image.jpg";
	string filename = "../data/overexposed.jpg";
	cout << "Filename: " << filename << endl;

	Mat img = imread(filename);

	if (img.cols > img.rows)
		resize(img, img, Size(500, img.rows*500/img.cols));
	else 
		resize(img, img, Size(img.cols*500/img.rows, 500));
	

	int num_channels = img.channels();
	vector<Mat> img_channels(num_channels);


	// Splitting the image in it's three channels
	split(img, img_channels);

	// Histogram Parameters
	vector<Mat> hists(num_channels);
	vector<int> channels = {0}, histSize = {256};
	vector<float> ranges = {0, 256}; //ranges of b,g and r

	// Histogram computation
	for(int i=0; i<num_channels; i++)
		calcHist(vector<Mat>{img_channels[i]}, channels, Mat(), hists[i], 
				histSize, ranges);
	
	
	// Showing original image and it's channels
	cout << "Original Image and it's channels:\n";	
	namedWindow(GENERAL_WINDOW);
	imshow_hists(GENERAL_WINDOW, img, hists);
	waitKey();
	
	// Equalizing image
	Mat eq_img;

	// Equalizing every channel's histogram ..
	for(int i=0; i<num_channels; i++) 
		equalizeHist(img_channels[i], img_channels[i]);	
	// .. and merging in the original image
	merge(img_channels, eq_img);

	// Hisogram computation
	for(int i=0; i<num_channels; i++)
		calcHist(vector<Mat>{img_channels[i]}, channels, Mat(), hists[i], 
				histSize, ranges);
	
	// Showing equalized image and it's channels
	cout << "Equalized Image and it's channels:\n\n";
	imshow(GENERAL_WINDOW, eq_img);
	imshow_hists(GENERAL_WINDOW, eq_img, hists);
	waitKey();


	// Equalizing in HSV
	Mat hsv_img;
	cvtColor(img, hsv_img, COLOR_BGR2HSV);

	for(int i=0; i<num_channels; i++){
		// Splitting the image in it's three channels, ..
		split(hsv_img, img_channels);
		// .. equalizing..
		equalizeHist(img_channels[i], img_channels[i]);
		// .. and merging together
		merge(img_channels, eq_img);
	
		// back to BGR colorspace to show
		cvtColor(eq_img, eq_img, COLOR_HSV2BGR);

		// Showing difference between images
		cout << "Difference between original and equalized (only ";	
		if (i==0) cout << "hue channel):\n";
		if (i==1) cout << "saturation channel):\n";
		if (i==2) cout << "value channel):\n";
		
		imshow_sided(GENERAL_WINDOW, img, eq_img);
		waitKey();
	}
	destroyAllWindows();
	

	// --- PART2: IMAGE FILTERING ---
	
	// preparing some cv::Mat for the different filters
	Mat median_img, gaussian_img, bilateral_img;
	median_img = gaussian_img = bilateral_img = eq_img.clone();
	
	
	// Median Filter
	int kernel_size = 1;
	MedianFilter mf = MedianFilter(median_img, kernel_size);
	
	cout << "\nMove the trackbars to apply a median filter, " <<
		 "anykey to pass on gaussian blur" << endl;
	
	// prepare window and trackbar
	namedWindow(MEDIAN_FILTER_WINDOW);
	createTrackbar("Kernel size",  			// trackbar name
			MEDIAN_FILTER_WINDOW, 			// window name
			&kernel_size, 					// addr of trackbar param
			30, 							// max filter size
			onMedianFilterKernelSize, 		// callback
			static_cast<void*>(&mf));		// data
	
	// done this because it fix the window size
	imshow(MEDIAN_FILTER_WINDOW, median_img);
	waitKey(0);  
	destroyAllWindows();
	

	// Gaussian Blur
	int sigma_int = 1;
	kernel_size = 1;

	GaussianFilter gf = GaussianFilter(gaussian_img, kernel_size, 
			  sigma_int/10.0); //10.0 is a possible int->double ''conversion'

	cout << "\nMove the trackbars to apply a guassian blur," << 
		 "anykey to pass on bilateral filtering" << endl;

	// prepare window and trackbar
	namedWindow(GAUSSIAN_FILTER_WINDOW);
	
	// kernel size trackbar
	createTrackbar("Kernel size",			// trackbar name
			GAUSSIAN_FILTER_WINDOW,			// window name 
			&kernel_size, 					// addr of trackbar param
			30, 							// max filter size
			onGaussianBlurKernelSize, 		// callback
			static_cast<void*>(&gf));		// data
	
	// sigma trackbar
	createTrackbar("Sigma", 				// trackbar name
			GAUSSIAN_FILTER_WINDOW, 		// window name
			&sigma_int,						// addr of trackbar param 
			300, 							// max sigma size
			onGaussianBlurSigma, 			// callback
			static_cast<void*>(&gf));		// data
	
	// done this because it fix the window size
	imshow(GAUSSIAN_FILTER_WINDOW, gaussian_img);
	waitKey(0);
	destroyAllWindows();
	

	// Bilateral Filter
	int sigma_range, sigma_space;
	sigma_range = sigma_space = 1;

	BilateralFilter bf = BilateralFilter(bilateral_img, 
			  sigma_range, sigma_space);

	cout << "\nMove the trackbars to apply a guassian blur," << 
		"anykey to end program" << endl;

	// prepare window and trackbar
	namedWindow(BILATERAL_FILTER_WINDOW);
	
	// sigma range trackbar
	createTrackbar("Sigma range",			// trackbar name 
			BILATERAL_FILTER_WINDOW, 		// window name
			&sigma_range, 					// addr of trackbar param
			300, 							// max srange (can be high)
			onBilateralFilterSigmaRange, 	// callback
			static_cast<void*>(&bf));		// data
	
	// sigma space trackbar
	createTrackbar("Sigma space",			// trackbar name 
			BILATERAL_FILTER_WINDOW, 		// window name
			&sigma_space, 					// addr of trackbar param
			5,  							// max sspace (low for comp issues)
			onBilateralFilterSigmaSpace, 	// callback
			static_cast<void*>(&bf));		// data
	
	// done this because it fix the window size
	imshow(BILATERAL_FILTER_WINDOW, bilateral_img);
	waitKey(0);
	

	return 0;
}

///////////////////////////////////////////////////////////////////////////////
// STL methods, functions and containers -> snake_case
// OpenCV & custom functions -> camelCase
// variables -> snake_case
