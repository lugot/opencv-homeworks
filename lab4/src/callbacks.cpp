#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../include/global.h"
#include "../include/filter.h"
#include "../include/detector.h"
#include "../include/utils.h"

// Canny Detector
void onEdgesDetectorMinThreshold(int position, void *userdata){
	EdgesDetector* cd = static_cast<EdgesDetector*>(userdata);
	
	cd->setMinThreshold(position*1.0);
	cd->doDetection();

    cv::Mat res;
    cv::cvtColor(cd->getResult(), res, cv::COLOR_GRAY2BGR);
	imshowSided(EDGE_DETECTOR_WINDOW, REFERENCE_IMG, res);
}
void onEdgesDetectorMaxThreshold(int position, void *userdata){
	EdgesDetector* cd = static_cast<EdgesDetector*>(userdata);
	
	cd->setMaxThreshold(position*1.0);

	cd->doDetection();

    cv::Mat res;
    cv::cvtColor(cd->getResult(), res, cv::COLOR_GRAY2BGR);
	imshowSided(EDGE_DETECTOR_WINDOW, REFERENCE_IMG, res);
}


// Hough Transform Lines
void onLinesDetectorRho(int position, void* userdata){
	LinesDetector* ld = static_cast<LinesDetector*>(userdata);
	
	if (position == 0) position++;
	ld->setRho(position/100.0); //range go [0..3] px

	ld->doDetection();
	std::vector<cv::Vec3f> lines = ld->getResult();

	cv::Mat res = REFERENCE_IMG.clone(); 
	drawLinesOnImage(res, lines);
    cv::resize(res, res, adaptedSize(res));
    imshow(LINES_DETECTOR_WINDOW, res);
}
void onLinesDetectorTheta(int position, void* userdata){
	LinesDetector* ld = static_cast<LinesDetector*>(userdata);
   
    if (position == 0) position++;
	ld->setTheta((CV_PI/180)*position/100.0); //range goes [0..5] degree

	ld->doDetection();
	std::vector<cv::Vec3f> lines = ld->getResult();

	cv::Mat res = REFERENCE_IMG.clone(); 
	drawLinesOnImage(res, lines);
    cv::resize(res, res, adaptedSize(res));
    imshow(LINES_DETECTOR_WINDOW, res);
}
void onLinesDetectorThreshold(int position, void* userdata){
	LinesDetector* ld = static_cast<LinesDetector*>(userdata);

	ld->setThreshold(position);

	ld->doDetection();
	std::vector<cv::Vec3f> lines = ld->getResult();

	cv::Mat res = REFERENCE_IMG.clone(); 
	drawLinesOnImage(res, lines);
    cv::resize(res, res, adaptedSize(res));
    imshow(LINES_DETECTOR_WINDOW, res);
}

// Hough Transform Circles
void onCirclesDetectorDp(int position, void *userdata){
	CirclesDetector* cd = static_cast<CirclesDetector*>(userdata);
	
	if (position == 0) position++;
	cd->setDp(position/10.0);

	cd->doDetection();
	std::vector<cv::Vec4f> circles = cd->getResult();

	cv::Mat res = REFERENCE_IMG.clone(); 
	drawCirclesOnImage(res, circles);
    cv::resize(res, res, adaptedSize(res));
    imshow(CIRCLES_DETECTOR_WINDOW, res);
}
void onCirclesDetectorMinDist(int position, void *userdata){
	CirclesDetector* cd = static_cast<CirclesDetector*>(userdata);
	
    if (position == 0) position++;
	cd->setMinDist(position*1.0);

	cd->doDetection();
	std::vector<cv::Vec4f> circles = cd->getResult();

	cv::Mat res = REFERENCE_IMG.clone(); 
	drawCirclesOnImage(res, circles);
    cv::resize(res, res, adaptedSize(res));
    imshow(CIRCLES_DETECTOR_WINDOW, res);
}
void onCirclesDetectorCannyThreshold(int position, void *userdata){
	CirclesDetector* cd = static_cast<CirclesDetector*>(userdata);
	
	if (position == 0) position++;
	cd->setCannyThreshold(position*1.0);

	cd->doDetection();
	std::vector<cv::Vec4f> circles = cd->getResult();

	cv::Mat res = REFERENCE_IMG.clone(); 
	drawCirclesOnImage(res, circles);
    cv::resize(res, res, adaptedSize(res));
    imshow(CIRCLES_DETECTOR_WINDOW, res);
}
void onCirclesDetectorAccumulatorThreshold(int position, void *userdata){
	CirclesDetector* cd = static_cast<CirclesDetector*>(userdata);
	
	if (position == 0) position++;
	cd->setAccumulatorThreshold(position*1.0);

	cd->doDetection();
	std::vector<cv::Vec4f> circles = cd->getResult();

	cv::Mat res = REFERENCE_IMG.clone(); 
	drawCirclesOnImage(res, circles);
    cv::resize(res, res, adaptedSize(res));
    imshow(CIRCLES_DETECTOR_WINDOW, res);
}
