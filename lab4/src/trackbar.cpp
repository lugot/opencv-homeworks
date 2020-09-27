#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
 
#include "../include/global.h"
#include "../include/filter.h"
#include "../include/utils.h"
#include "../include/callbacks.h"
#include "../include/trackbar.h"

EdgesDetector edgesDetectorSelector(cv::Mat img){

    int min_threshold, max_threshold;
    cv::Mat tmp_img = img.clone(); //used for first visualisation

    // computing default parameters
    min_threshold = 342; //use this for final result
    max_threshold = 366; 

	EdgesDetector ed = EdgesDetector(img, 342, 366);

    std::cout << "Move the trackbars to detect edges with Canny algorithm, " << 
        "the parameters are:\n" <<
        "\tMin threshold: the first threshold is the min\n" << 
        "\tMax threshold: the second threshold is the max " << 
        "\nAnykey to move to Lines Detector." << std::endl;

    cv::namedWindow(EDGE_DETECTOR_WINDOW);
	// Max Threshold trackbar 
    cv::createTrackbar("Min threshold",  			 
			EDGE_DETECTOR_WINDOW,		 			 
			&min_threshold, 					 
			600, //extreme range							 
			onEdgesDetectorMinThreshold, 		 
			static_cast<void*>(&ed));		
	
	// Max Threshold trackbar 
    cv::createTrackbar("Max threshold",  			 
			EDGE_DETECTOR_WINDOW,		 			 
			&max_threshold, 					 
			600, //extreme range		 
			onEdgesDetectorMaxThreshold, 		
			static_cast<void*>(&ed));	

    ed.doDetection();
    cv::cvtColor(ed.getResult(), tmp_img, cv::COLOR_GRAY2BGR);
    
    imshowSided(EDGE_DETECTOR_WINDOW, REFERENCE_IMG, tmp_img); 
    cv::waitKey(); 
    cv::destroyAllWindows();
        
    return ed;
}

LinesDetector linesDetectorSelector(cv::Mat img){

	int rho, theta, threshold;
	rho = 100; theta = 100; threshold = 75;

	// default parameters
    LinesDetector ld = LinesDetector(img, 1, CV_PI/180, 75);

    std::cout << "Move the trackbars to detect lines with Hough Detector, " <<
		"the parameters are:\n" <<
        "\t      Rho: resolution of rho in pixel (range in [0..3] px, suggested 1.0)\n" <<
        "\t    Theta: resolution of theta (range in [0..5] degrees, suggested 1 degree)\n" <<
        "\tThreshold: minimum number of intersection to detect a line" << 
        "\nAnykey to stop and move to circles detector" << std::endl;
	
	// prepare window and trackbar
    cv::namedWindow(LINES_DETECTOR_WINDOW);

	// Rho trackbar 
    cv::createTrackbar("Rho",
			LINES_DETECTOR_WINDOW,
			&rho, 
			300,					
			onLinesDetectorRho,
			static_cast<void*>(&ld));
	
	// Theta trackbar 
    cv::createTrackbar("Theta",
			LINES_DETECTOR_WINDOW,
			&theta,
			500,
			onLinesDetectorTheta,
			static_cast<void*>(&ld));
	
	// Threshold trackbar
    cv::createTrackbar("Threshold",
			LINES_DETECTOR_WINDOW,
			&threshold,
			500,
			onLinesDetectorThreshold,
			static_cast<void*>(&ld));

    cv::Mat res = REFERENCE_IMG.clone();
    ld.doDetection();
    drawLinesOnImage(res, ld.getResult());
    resize(res, res, adaptedSize(res));

    cv::imshow(LINES_DETECTOR_WINDOW, res);
    cv::waitKey(); 
    cv::destroyAllWindows();
    
    return ld;
}

CirclesDetector circlesDetectorSelector(cv::Mat img){
	
	int dp, min_dist, canny_thresh, acc_tresh;
	dp = 10; min_dist = 100; canny_thresh = 630; acc_tresh = 35;

	CirclesDetector cd = CirclesDetector(img, 1.0, 100, 630, 30);

    std::cout << "Move the trackbars to detect circles with Hough Transform, " <<
		"the parameters are:\n" << 
        "\t      Inverse ratio treshold: keep in range [0.8..1.5] (actual range is [0..5])\n" <<
        "\tMinimum dist between centers: keep proportional to rows (range [0..1000])\n" <<
        "\t             Canny threshold: max threshold during canny edges detection\n" <<
        "\t        Accumulator threshold: to tune based on false positive\n" <<
        "Anykey to  stop and compute the final image." << std::endl;
	
	// prepare window and trackbar
    cv::namedWindow(CIRCLES_DETECTOR_WINDOW);

    // Inverse ratio treshold
    cv::createTrackbar("Inverse ratio threshold",
			CIRCLES_DETECTOR_WINDOW,
			&dp, 
			50,					
			onCirclesDetectorDp,
			static_cast<void*>(&cd));
	
	// Minimum dist btw centers trackbar 
    cv::createTrackbar("Min center dist",
			CIRCLES_DETECTOR_WINDOW,
			&min_dist, 
			1000,					
			onCirclesDetectorMinDist,
			static_cast<void*>(&cd));
	
	// Canny threshold trackbar 
    cv::createTrackbar("Canny Threshold",
			CIRCLES_DETECTOR_WINDOW,
			&canny_thresh, 
			1000,					
			onCirclesDetectorCannyThreshold,
			static_cast<void*>(&cd));

	//  Accumulator threshold trackbar
    cv::createTrackbar("Accumulator trehshold",
			CIRCLES_DETECTOR_WINDOW,
			&acc_tresh, 
			300,					
			onCirclesDetectorAccumulatorThreshold,
			static_cast<void*>(&cd));
    
    cv::Mat res = REFERENCE_IMG.clone();
    cd.doDetection();
    drawCirclesOnImage(res, cd.getResult());
    resize(res, res, adaptedSize(res));

    cv::imshow(CIRCLES_DETECTOR_WINDOW, res);
    cv::waitKey(); 
    cv::destroyAllWindows();

    return cd;
}
