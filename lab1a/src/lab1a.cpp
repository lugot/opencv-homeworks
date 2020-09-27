//OpenImg.cc
#include<bits/stdc++.h>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

#define NEIGHBORHOOD 9

using namespace std;
using namespace cv;

void onMouse(int event, int x, int y, int f, void* userdata);
void tuneThreshold(Mat &img, int x, int y);
int thresholdCounter(Mat &img, int threshold_value, Vec3b color);
Rect adaptRect(Mat &img, int x, int y);

bool threshold_tuned = false;
Mat mask;
int pixel_mask, threshold_value;

int main(int argc, char** argv){
	Mat img = imread("../data/robocup.jpg");
	resize(img, img, Size(img.cols/1.3, img.rows/1.3));

	namedWindow("lab1a");
	imshow("lab1a", img);
	
	setMouseCallback("lab1a", onMouse, (void*)&img);
	waitKey(0);

	return 0;
}

void onMouse(int event, int x, int y, int f, void* userdata){
	if (event == EVENT_LBUTTONDOWN){
		cout << "Found EVENT_LBUTTONDOW on (" << x << "," << y << ")\n";
		
		Mat img = *(Mat*) userdata, out=img.clone();
		if (!threshold_tuned) tuneThreshold(img, x, y);	
		
		Scalar mean = cv::mean(img(adaptRect(img, x, y)));
		cout << "Mean: " << mean << endl;
			
		int pixel_counter = 0;
		for(int i=0; i<out.rows; i++) for(int j=0; j<out.cols; j++){
			if ( fabs(out.at<Vec3b>(i,j)[0] - mean[0]) < threshold_value*2 and
				 fabs(out.at<Vec3b>(i,j)[1] - mean[1]) < threshold_value*2 and
				 fabs(out.at<Vec3b>(i,j)[2] - mean[2]) < threshold_value*2 and
				 mask.at<Vec3b>(i,j) != Vec3b({0,0,0})){
				pixel_counter++;
			}	
		}

		if (pixel_counter > pixel_mask*30/100){
			for(int i=0; i<out.rows; i++) for(int j=0; j<out.cols; j++){
				if ( fabs(out.at<Vec3b>(i,j)[0] - mean[0]) < threshold_value*2 and
					 fabs(out.at<Vec3b>(i,j)[1] - mean[1]) < threshold_value*2 and
					 fabs(out.at<Vec3b>(i,j)[2] - mean[2]) < threshold_value*2 and
					 mask.at<Vec3b>(i,j) != Vec3b({0,0,0})){
					
					out.at<Vec3b>(i,j)[0] = 201;
					out.at<Vec3b>(i,j)[1] = 37;
					out.at<Vec3b>(i,j)[2] = 92;
				}	
			}


		imshow("lab1a", out);
		}
	}
	return;
}

void tuneThreshold(Mat &img, int x, int y){
	
	Mat mod;
	threshold(img, mod, 64, 256, THRESH_BINARY); //64 is 'middle' value
	Vec3b color = mod.at<Vec3b>(y,x);

	int lower=25, upper=100, mid;
	
	int lower_px = thresholdCounter(img, lower, color),
		upper_px = thresholdCounter(img, upper, color),
		mid_px;

	while(lower < upper){ //binary search to speed-up
		mid = (upper+lower)/2;
		mid_px = thresholdCounter(img, mid, color);
	
		if (mid_px > lower_px and mid_px > upper_px){ //not crescent, more cases
			if (lower_px > upper_px){
				upper=mid;
				upper_px=mid_px;
			}else{
				lower=mid;
				lower_px=mid_px;
			}
		}
		else if (mid_px > lower_px){
			lower=mid;
			lower_px=mid_px;
		}
		else{
			upper=mid;
			upper_px=mid_px;
		} 				   
	}

	cout << "Generating mask..\n";
	
	threshold(img, mask, mid, 256, THRESH_BINARY);
	for(int i=0; i<mask.rows; i++) for(int j=0; j<mask.cols; j++){
		if (mask.at<Vec3b>(i,j) != color) mask.at<Vec3b>(i,j) = {0,0,0};
	}
	pixel_mask=mid_px;
	threshold_value=mid;

	cout << "\t threshold: " << mid <<
			"\n\tnum pixels: " << mid_px << endl << endl;

	//cout << "Visualizing mask..\n";
	//imshow("lab1a", mask);
	//waitKey(0);

	threshold_tuned=true;
	return;
}

int thresholdCounter(Mat &img, int threshold_value, Vec3b color){
	Mat mod;
	threshold(img, mod, threshold_value, 256, THRESH_BINARY);

	int counter=0;
	for(int i=0; i<mod.rows; i++) for(int j=0; j<mod.cols; j++)
		if (mod.at<Vec3b>(i,j) == color) counter++;
	
	return counter;
}
Rect adaptRect(Mat &img, int x, int y){
	int topx = max(x-NEIGHBORHOOD, 0),
		topy = max(y-NEIGHBORHOOD, 0);
	int neighborhood_size = min(NEIGHBORHOOD, img.cols-x);
		neighborhood_size = min(neighborhood_size, img.rows-y);

	cout << "Building rectangle starting from " <<
		"(" << topx << ","<< topy << ")" <<
		" of dimension " << neighborhood_size << endl;
		
	return Rect(topx, topy, neighborhood_size, neighborhood_size);
}
