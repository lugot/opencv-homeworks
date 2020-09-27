#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/base.hpp>
#include <opencv4/opencv2/core/fast_math.hpp>
#include <opencv4/opencv2/core/matx.hpp>
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

using namespace std;
using namespace cv;

// defining global flag
int DEBUG;
int SCREEN_HEIGHT;
int SCREEN_WIDTH;
Mat REFERENCE_IMG;
    
int main(int argc, char** argv){

    // debug parsing
    if (argc > 1) DEBUG = stoi(string(argv[1]));
    else DEBUG = 0; //default
    
    if (DEBUG) cout << "DEBUG mode: more infos will be printed\n"; 
    
    // screen size parsing
    if (argc > 3){
        SCREEN_HEIGHT = stoi(string(argv[5]));
        SCREEN_WIDTH = stoi(string(argv[6]));
    } else { //default
        SCREEN_HEIGHT = 500; 
        SCREEN_WIDTH = 1200;
    }

	// Loading image
	String image_path;
    cout << "Path of the image: \n"<< 
        "(0: default: data/input.png) > ";
    cin >> image_path;

    // chech if default is provided
    if (image_path.length() == 1 and image_path[0] == '0')
        image_path = "data/input.png";

	Mat img = imread("../" + image_path), canny_img;
    if (img.data == NULL){
        cout << "No image found! Aborted.\n";
        return 1;
    }
    // saving project-wide version
    REFERENCE_IMG = img.clone();



	// --- PART1: LINES DETECTION ---
    
    cout << "PART1: Lines detection\n\n";
    
    // preparing gray level version
	cvtColor(img, canny_img, COLOR_BGR2GRAY);
    // blurring: this actually helps a lot, it keeps the thresholds low
    GaussianFilter gf = GaussianFilter(canny_img, 5, 4.0);
    gf.doFilter(); 
    canny_img = gf.getResult();

    // first: edges detection
    EdgesDetector ed = edgesDetectorSelector(canny_img);	
    // second: lines detection
    LinesDetector ld = linesDetectorSelector(ed.getResult());	



	// --- PART2: HOUGH CIRCLES ---
	
    cout << "\n\nPART2: Circles detection\n\n";
    
    // preparing gray level version
	cvtColor(img, canny_img, COLOR_BGR2GRAY);
    // one shot circles detector (no blur needed) 
    CirclesDetector cd = circlesDetectorSelector(canny_img);
   


    // --- PART3: FINAL DRAWING ---
    
    // if not enought elements detected abort
    if (ld.getResult().size() < 2 or cd.getResult().size() < 1){
        cout << "\n\nDetection Failed! Please retune the parameters." << endl;
        return 0;
    }

    vector<Vec3f> lines = ld.getResult();
    Vec3f line1, line2;

    auto max_line_iter = std::max_element(lines.begin(), lines.end(),
            [](const Vec3f l1, const Vec3f l2) -> bool {
                return l1[2] < l2[2];
            }); 

    line1 = *max_line_iter;
    lines.erase(max_line_iter);
    line2 = *std::max_element(lines.begin(), lines.end(),
            [](const Vec3f l1, const Vec3f l2) -> bool {
                return l1[2] < l2[2];
            }); 

    // Triangle drawing
    Point p1, p2, p3;
    double theta, rho, a, b, x0, y0;

    double m1, m2, q1, q2;
    m1 = -1.0/tan(line1[1]);
    m2 = -1.0/tan(line2[1]);
    q1 = line1[0]*(sin(line1[1]) + cos(line1[1])*cos(line1[1])/sin(line1[1])); 
    q2 = line2[0]*(sin(line2[1]) + cos(line2[1])*cos(line2[1])/sin(line2[1])); 
    
    p1.x = (img.rows-q1)/m1;
    p1.y = img.rows;

    p2.x = (img.rows-q2)/m2;
    p2.y = img.rows;
    //intersection between lines
    p3.x = (q2-q1)/(m1-m2);
    p3.y = m1*(q2-q1)/(m1-m2) + q1;

    vector<vector<Point>> poly;
    poly.push_back({p1, p2, p3});

    // drawing triangle
    fillPoly(img, poly, Scalar(0,0,255));


    // Circle drawing
    vector<Vec4f> circles = cd.getResult();
    Vec4f circle = *std::max_element(circles.begin(), circles.end(),
            [](const Vec4f c1, const Vec4f c2) -> bool {
                return c1[3] < c2[3];
            });
       
    Point center = Point(cvRound(circle[0]), cvRound(circle[1]));
    int radius = cvRound(circle[2]) +1;

    // circle drawing
    cv::circle(img, center, radius, Scalar(0,255,0), -1, 8, 0);


    // show image
    namedWindow(GENERAL_WINDOW);
    imshow(GENERAL_WINDOW, img);
    waitKey();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// STL methods, functions and containers -> snake_case
// OpenCV & custom functions -> camelCase
// variables -> snake_case
