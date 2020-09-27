#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

void imshow_sided(std::string windowname, cv::Mat img, cv::Mat img2){
	int width = 500;	
	resize( img,  img, cv::Size(width, img.rows*width/img.cols));
	resize(img2, img2, cv::Size(width, img.rows*width/img.cols));
	
	cv::Mat tiled = cv::Mat::zeros(img.rows, width*2, img.type());
	img.copyTo(tiled.rowRange(0, img.rows).colRange(0, width));
	img2.copyTo(tiled.rowRange(0, img.rows).colRange(width, width*2));

	imshow(windowname, tiled);
}

void imshow_hists(std::string windowname, cv::Mat img, 
		std::vector<cv::Mat>& hists){
	
	// Code from provided function showHistogram
	double hmax[3] = {0,0,0};
	double min_val;
	minMaxLoc(hists[0], &min_val, &hmax[0]);
	minMaxLoc(hists[1], &min_val, &hmax[1]);
	minMaxLoc(hists[2], &min_val, &hmax[2]);

	std::string wname[3] = { "blue", "green", "red" };
	cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0), 
		cv::Scalar(0,0,255) };

	std::vector<cv::Mat> canvas(hists.size());
	for (int i = 0, end = hists.size(); i < end; i++){
    	canvas[i] = cv::Mat::ones(128, hists[0].rows, CV_8UC3);

		for (int j = 0, rows = canvas[i].rows; j < hists[0].rows-1; j++){
			line(canvas[i],
				cv::Point(j, rows),
        		cv::Point(j, rows - (hists[i].at<float>(j) * rows/hmax[i])),
					hists.size() == 1 ? cv::Scalar(200,200,200) : colors[i],
        			1, 8, 0
    				);
    	}
	}
	
	resize(img, img, cv::Size(256*3, img.rows*(256*3)/img.cols));
	
	cv::Mat tiled = cv::Mat::zeros(128+img.rows, 256*3, img.type());
	
	canvas[0].copyTo(tiled.rowRange(0,128).colRange(0,256));
	canvas[1].copyTo(tiled.rowRange(0,128).colRange(256,512));
	canvas[2].copyTo(tiled.rowRange(0,128).colRange(512,768));
	img.copyTo(tiled.rowRange(128, tiled.rows));

	imshow(windowname, tiled);
}
