#include<bits/stdc++.h>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv4/opencv2/videoio.hpp>

#define NEIGHBORHOOD 9
#define THRESHOLD 5

using namespace std;
using namespace cv;

void onMouse(int event, int x, int y, int f, void* userdata);

Mat out;

int main(int argc, char** argv){
	
	int i = 3;


	// Image in argument
	Mat img = imread("../data/robocup.jpg");
	resize(img, img, Size(img.cols/2.0, img.rows/2.0));
	out = img.clone();
	
	cout << img.channels() << " " << img.depth() << endl;

	cout << img.at<Vec2b>(0,0) << endl << img.at<Vec3b>(0,0) << 
		" " << img.at<Vec3f>(0,0) << endl;
	
	namedWindow("lab1");
	imshow("lab1", img);
			
	
	setMouseCallback("lab1", onMouse, (void*)&img);
	waitKey(0);

	return 0;
}

void onMouse(int event, int x, int y, int f, void* userdata){
	if (event == EVENT_LBUTTONDOWN){
		cout << "Found EVENT_LBUTTONDOW on (" << x << "," << y << ")\n";
		
		Mat img = *(Mat*) userdata, rectHSV, out = img.clone();
		
		int topx = max(x-NEIGHBORHOOD, 0),
			topy = max(y-NEIGHBORHOOD, 0);
		int neighborhood_size = min(NEIGHBORHOOD, img.cols-x);
			neighborhood_size = min(neighborhood_size, img.rows-y);

		cout << "Building rectangle starting from " <<
			"(" << topx << ","<< topy << ")" <<
			" of dimension " << neighborhood_size << endl;
		Rect rect(topx, topy, neighborhood_size, neighborhood_size);

		cvtColor(img(rect), rectHSV, COLOR_RGB2HSV);

		Scalar mean = cv::mean(rectHSV);
				
		cout << "mean: " << mean << endl;
		
		cvtColor(img, out, COLOR_RGB2HSV);
		for(int i=0; i<out.rows; i++) for(int j=0; j<out.cols; j++){
		
			/*
			if (j==out.cols-1) cout << "Stop per morire" <<endl;

			Mat pixelHSV;
			cvtColor(out(Rect(i,j,1,1)),
					pixelHSV,
					COLOR_RGB2HSV);

			if (fabs(pixelHSV.at<Vec3b>(0,0)[0] - mean[0]) < THRESHOLD){
				pixelHSV.at<Vec3b>(0,0)[0] += 30;
				
				Mat pixelRGB;
				cvtColor(pixelHSV, pixelRGB, COLOR_HSV2RGB);
				out.at<Vec3b>(i,j) = pixelRGB;
			}
			*/

			if (fabs(out.at<Vec3b>(i,j)[0] - mean[0]) < THRESHOLD){
				out.at<Vec3b>(i,j)[0] +=30;
			}
		}
		
		cvtColor(out, out, COLOR_HSV2RGB);
		imshow("lab1", out);
	

	}
	return;
}

///////////////////////////////////////////////////////////////////////////////
