#include <bits/stdc++.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv4/opencv2/core.hpp>

// i don't know if this is the right way to do that, but my
//  linter gets mad if I do differently
#include "../include/global.h"
#include "../include/utils.h"

using namespace std;
using namespace cv;

int DEBUG;
int SCREEN_HEIGHT;
int SCREEN_WIDTH;

int main(int argc, char** argv){
    
    // debug parsing
    if (argc > 1) DEBUG = stoi(string(argv[1]));
    else DEBUG = 0; //default
    
    if (DEBUG) cout << "DEBUG mode: press a key to skip between checkerboard photos\n\n";
    
    // screen size parsing
    if (argc > 3){
        SCREEN_HEIGHT = stoi(string(argv[5]));
        SCREEN_WIDTH = stoi(string(argv[6]));
    } else { //default
        SCREEN_HEIGHT = 500; 
        SCREEN_WIDTH = 1000;
    }
    
    std::vector<int> a;
      
    // checkerboard directory parsing
    String checkerboard_directory;
    
    cout << "Checkerboard photos directory, only png extension: \n" << 
        "(0: default: data/checkerboard_images) > ";
    cin >> checkerboard_directory;
    
    // check if default value is provided
    if (checkerboard_directory.length() == 1 and checkerboard_directory[0] == '0')    
        checkerboard_directory = "data/checkerboard_images"; // default value

    // harvesting images
    vector<String> filenames;
    try {
        utils::fs::glob("../" + checkerboard_directory, "*.png", filenames, false, false);
    } catch (cv::Exception) {
        cout << "Not enough images! Aborted.\n";
        return 1;
    }
    cout << "Loaded " << filenames.size() << " images from " << checkerboard_directory << endl;


    // checkerboard proprieties parsing
    int checkerboard_rows, checkerboard_cols;
    double square_size;
    
    cout << "\nCheckerboard proprieties (number of internal rows, cols and square size):\n" << 
        "(0 0 0: default: 5 6 0.11) > ";
    cin >> checkerboard_rows >> checkerboard_cols >> square_size;

    if (checkerboard_rows == 0 && checkerboard_cols == 0 && square_size == 0){
        checkerboard_rows = 5;
        checkerboard_cols = 6;
        square_size = 0.11;
    }

    if (checkerboard_rows <= 0 or checkerboard_cols <= 0 or square_size <= 0){
        cout << "Incorrect parameters! Aborted.";
        return 1;
    }



    // --- PART 1: CHECKERBOARD CORNERS DETECTION --- 

    vector<vector<Point3f>> checkerboard_points;
    vector<vector<Point2f>> image_points;
    int desired_corners = checkerboard_rows*checkerboard_cols; 
    Mat img;
    
    cout << "\nCollecting images..\n";
    for(string &filename: filenames){
        img = imread(filename);
    
        // corners detection
        vector<Point2f> corners;
        findChessboardCorners(img, Size(checkerboard_cols, checkerboard_rows), corners);

        cout << "\tfound " << corners.size() << " corners in " << 
            filename.substr(28); // adapter to file location
        
        if (corners.size() != desired_corners) cout << " -> deleted.";
        else{
            // refine corner detection to subpixel
            Mat gray_img;
            double halfCornersDist = static_cast<int>(0.5*l2dist(corners[0], corners[1]));
            
            cvtColor(img, gray_img, COLOR_BGR2GRAY); // needed by subPix
            cornerSubPix(gray_img, corners, 
                    Size(halfCornersDist, halfCornersDist), 
                    Size(-1,-1), //unused 
                    TermCriteria(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03));
            
            // add to pool
            image_points.push_back(corners);
            
            // 3D points computation
            vector<Point3f> corners3D;
            find3DCorners(checkerboard_rows, checkerboard_cols, corners3D, square_size);
            checkerboard_points.push_back(corners3D);
        }
        cout << "\n";
    

        if (DEBUG){
            drawChessboardCorners(img, 
                    Size(checkerboard_cols,checkerboard_rows), 
                    corners, true); 
            resize(img, img, adaptedSize(img));
            namedWindow(GENERAL_WINDOW);
            imshow(GENERAL_WINDOW, img);
            waitKey();
        }   
    }
    destroyAllWindows(); //prevent additional keypressing
    cout << "Computed " << checkerboard_points.size() << " images\n\n";
    
    if (checkerboard_points.size() == 0){
        cout << "Not enough corners! Aborted.\n";
        return 1;
    }


    // --- PART 2: CAMERA CALIBRATION  --- 
    
    cout << "\nCalibration..." << flush;
    Mat camera_matrix = Mat::zeros(8, 1, CV_64F);
    vector<double> dist_coeffs;
    vector<Mat> rvecs, tvecs;

    // calibration
    auto start = std::chrono::high_resolution_clock::now();
    
    calibrateCamera(checkerboard_points, image_points, 
            Size(checkerboard_cols, checkerboard_rows),
            camera_matrix, dist_coeffs, rvecs, tvecs);
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
            (std::chrono::high_resolution_clock::now() - start);

    cout << " done! (in " << duration.count()/1000.0 << " msec)\n"; 


    // Collect camera parameters
    double fx=camera_matrix.at<double>(0,0),
           fy=camera_matrix.at<double>(1,1),
           cx=camera_matrix.at<double>(0,2),
           cy=camera_matrix.at<double>(1,2);
    
    double k1=dist_coeffs[0],
           k2=dist_coeffs[1],
           k3=dist_coeffs[4],
           p1=dist_coeffs[2],
           p2=dist_coeffs[3];


    // Print camera paramters
    cout << "\nCamera parameters:\n1. Camera matrix:\n";
    for(int i=0; i<3; i++){
        cout << "\t";
        for(int j=0; j<3; j++) cout << setw(7) << camera_matrix.at<double>(i,j) << " ";
        cout << "\n";
    }
    cout << "\n\tin particular:\n";
    cout << "\t- focal length in pixel (wrt x coord): " << fx << "\n";
    cout << "\t- focal length in pixel (wrt y coord): " << fy << "\n";
    cout << "\t- principal point in pixel: (" << cx << "," << cy << ")\n";
    
    cout << "\n2. Distortion parameters:\n";
    cout << "\tradial distortion parameters:\n";
    cout << "\t- k1: " << k1 << "\n"; 
    cout << "\t- k2: " << k2 << "\n"; 
    cout << "\t- k3: " << k3 << "\n"; 
    cout << "\ttangential distortion parameters:\n";
    cout << "\t- p1: " << p1 << "\n"; 
    cout << "\t- p2: " << p2 << "\n"; 
    
    

    // --- PART 3: REPROJECTION ERROR COMPUTATION --- 
        
    // Reprojection error computation
    cout << "\n\nReprojection and error computation.." << endl;
    vector<double> reprojection_errors;

    for(int i=0; i<checkerboard_points.size(); i++){
        vector<Point2f> projected_points;
        
        projectPoints(checkerboard_points[i], rvecs[i], tvecs[i], camera_matrix,
                dist_coeffs, projected_points);
        
        double dst=0.0;
        for(int j=0; j<projected_points.size(); j++){
            Point2f projected_point = projected_points[j],
                    image_point     = image_points[i][j]; 
            
            dst += l2dist(projected_point, image_point);
        }
        dst/=projected_points.size();
        
        reprojection_errors.push_back(dst);
    }

    int min_error_idx = min_element(reprojection_errors.begin(), 
            reprojection_errors.end()) - reprojection_errors.begin();
    int max_error_idx = max_element(reprojection_errors.begin(), 
            reprojection_errors.end()) - reprojection_errors.begin();

    cout << "\tBest calibration done on " << filenames[min_error_idx].substr(28) << 
        ": error=" << reprojection_errors[min_error_idx] << endl;   
    cout << "\tWorst calibration done on " << filenames[max_error_idx].substr(28) << 
        ": error=" << reprojection_errors[max_error_idx] << endl;

    

    // --- PART 4: RECTIFICATION --- 
    String image_path;
    
    cout << "\nPath of the image to rectify: \n" << 
        "(0: default: data/test_image.png) > ";
    cin >> image_path;
    
    // check if default value is provided
    if (image_path.length() == 1 and image_path[0] == '0')    
        image_path = "data/test_image.png"; // default value

    Mat original = imread("../" + image_path), rectified;
    if (original.data == NULL){
        cout << "No image found! Aborted.\n";
        return 1;
    }

    cout << "\nRectifing..." << flush;
    
    start = std::chrono::high_resolution_clock::now();
    
    Mat map1, map2;
    
    // computing new camera matrix..
    initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat::eye(3,3, CV_8U), 
            camera_matrix, original.size(), CV_32FC2, 
            map1, map2);
    
    // .. in order to rectify the image
    remap(original, rectified, map1, map2, INTER_NEAREST);
    
    duration = std::chrono::duration_cast<std::chrono::microseconds> 
            (std::chrono::high_resolution_clock::now() - start);

    cout << " done! (in " << duration.count()/1000.0 << " msec)\n"
        << "Original image on left side, rectified image on right side\n"; 
    
    resize(original, original, adaptedSize(original));
    resize(rectified, rectified, adaptedSize(original));
    imshowSided(GENERAL_WINDOW, original, rectified);
    waitKey();
    
    return 0;
}
///////////////////////////////////////////////////////////////////////////////
// STL methods, functions and containers -> snake_case
// OpenCV & custom functions -> camelCase
// variables -> snake_case
