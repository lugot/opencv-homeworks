#include <iostream>
#include <numeric>

#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/hal/interface.h>

#include "../include/global.h"
#include "../include/tracker.h"
//#include "global.h"
//#include "tracker.h"

using namespace std;
using namespace cv;

//Constructor
Tracker::Tracker(vector<Mat> objects, double ratio, bool draw_keypoints, bool draw_tracking){
   
    this->objects = objects; 
    this->ratio = ratio;
    this->draw_keypoints = draw_keypoints;
    this->draw_tracking = draw_tracking;

    // Reserve space for keypoints and descriptors forall objects
    obj_keypoints.reserve(objects.size());
    obj_descriptors.reserve(objects.size());
    
    // compute feature and descriptros forall objects
    for(Mat& obj: objects){
        vector<KeyPoint> keypoints;
        Mat descriptors;
        
        Tracker::computeFeatures(obj, keypoints, descriptors);

        obj_keypoints.push_back(keypoints);
        obj_descriptors.push_back(descriptors);
    }
    this->obj_corners = vector<vector<Point2f>>(objects.size());
    
    // Set random colors
    colors.resize(objects.size());
    for(Scalar& color: colors) color = Scalar(rand()%255, rand()%255, rand()%255);
    // default values for 4 objects
    if (colors.size() == 4){
        colors[0] = Scalar(0,0,255);
        colors[1] = Scalar(0,255,0);
        colors[2] = Scalar(255,0,0);
        colors[3] = Scalar(0,255,255);
    }
}

//Locate into the first image of the video a set of objects of interest, represented by an example image 
//(one example for each object). To locate objects, match the local features of the example images with 
//the input image

//Calculate Keypoints and Descriptors
void Tracker::computeFeatures(Mat img, vector<KeyPoint>& keypoints, Mat& descriptors){

   auto start = std::chrono::high_resolution_clock::now();
   
    Ptr<SIFT> SIFT_detector = SIFT::create();

    //Compute the keypoints and the descriptors for the input image (the first frame) 
    //and each one of the objects
    SIFT_detector -> detect(img, keypoints);
    SIFT_detector -> compute(img, keypoints, descriptors);
    
   
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
         (std::chrono::high_resolution_clock::now() - start);
   
    if (INFO){
      cout << "[INFO]: features computed in  " << duration.count()/1000.0 << " msec." << endl;
    }
    if (DEBUG){
        cout << "[DEBUG]: features extracted:" << endl; 
        Mat output;
        drawKeypoints(img, keypoints, output);
        resize(output, output, Tracker::adaptedSize(output));
        imshow(GENERAL_WINDOW, output);
        waitKey();
    }
}


//Calculate matching
void Tracker::matchFirstFrame(Mat first_frame, Mat& output_img){
    this->prev_frame = first_frame.clone(); 
    if (draw_tracking) tracking_mask = Mat::zeros(first_frame.size(), first_frame.type());

    output_img = first_frame.clone();
    
    // Computing feature and descriptor for first frame
    vector<KeyPoint> frame_keypoints;
    Mat frame_descriptors;
    
    Tracker::computeFeatures(first_frame, frame_keypoints, frame_descriptors);

    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2);
    vector<vector<DMatch>> matches = vector<vector<DMatch>>(objects.size());

    output_img = first_frame.clone();
   
    // Iterate allover objects to match features with first frame and save
    //  keypoints
    for(int i=0; i<objects.size(); i++){ 

        auto start = std::chrono::high_resolution_clock::now();

        // Matching from object to frame
        matcher->match(obj_descriptors[i], frame_descriptors, matches[i]); 

        // Compute the min distance in the matches
        int min_dist = INT_MAX;
        for(int j=0; j<matches[i].size(); j++) 
            min_dist = min(min_dist, (int) matches[i][j].distance);
        
        // Delete wrong matches (the ones that have distance > ratio * mindist)
        vector<DMatch> good_matches;
        for(int j=0; j<matches[i].size(); j++){
            if(matches[i][j].distance <= ratio*min_dist){
                good_matches.push_back(matches[i][j]);
            }
        }

        // Valid keypoints
        vector<Point2f> object_pts;
        vector<Point2f> frame_pts;
        // For every good match
        for(DMatch match: good_matches){
            object_pts.push_back( obj_keypoints[i][match.queryIdx].pt);
            frame_pts.push_back(frame_keypoints[match.trainIdx].pt);
        }

        if (object_pts.size() < 4 || frame_pts.size() < 4){ // Homography needs at least some points
            cout << "[ERROR]: tracking failed! Few points detected, retune the parameters" << endl;
            this->objects_pts[i].clear();
            continue;
        }
        
        Mat mask;
        // Perspective transformation between object and frame
        Mat H = findHomography(object_pts, frame_pts, mask, RANSAC);
        
        // Discard outliers
        for(int j=mask.rows-1; j>=0; j--){
            // 0 means outliers
            if (mask.at<uchar>(j,0) == 0) good_matches.erase(good_matches.begin()+j);
        }
    
        // Reserve and save points object points matched on the frame
        frame_pts.clear(); 
        frame_pts.reserve(good_matches.size()); 
        
        for(DMatch match: good_matches) frame_pts.push_back(frame_keypoints[match.trainIdx].pt);
        this->objects_pts.push_back(frame_pts);



        auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
             (std::chrono::high_resolution_clock::now() - start);
       
        if (INFO){
              cout << "[INFO]: features extraction + match (with Homography) on object " << i << " computed in "
                  << duration.count()/1000.0 << " msec." << endl;
        }
        if (DEBUG){
            cout << "[DEBUG]: matches on object " << i << endl;
            // Compute the resulting image:  left -> object to be detected
            //                              right -> frame                              
            Mat img_matches;
            drawMatches(objects[i], obj_keypoints[i], first_frame, frame_keypoints, 
                    good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

            namedWindow(GENERAL_WINDOW);
            resize(img_matches, img_matches, Tracker::adaptedSize(img_matches));
            imshow(GENERAL_WINDOW, img_matches);
            waitKey();
        }


        
        // Draw on output image
        for(Point2f pt: frame_pts) circle(output_img, pt, 3, colors[i], -1);
        drawRectangle(output_img, H, objects[i].size(), colors[i], i); 
    }
}


void Tracker::trackFrame(Mat next_frame, Mat& output_img){
   
    // Prepare two graylevel image for Lucas Kanade
    Mat prev_gray_frame, next_gray_frame;

    cvtColor(next_frame, next_gray_frame, COLOR_BGR2GRAY);
    cvtColor(prev_frame, prev_gray_frame, COLOR_BGR2GRAY);
       
    output_img = next_frame.clone();
    
    // Iterate over objects
    for(int i=0; i<this->objects_pts.size(); i++){

        if (this->objects_pts[i].size() == 0) continue;

        auto start = std::chrono::high_resolution_clock::now();

        vector<Point2f> new_object_pts; // for single object
        vector<uchar> status;
        vector<float> err;
        
        // Compute optical flow
        calcOpticalFlowPyrLK(prev_gray_frame, next_gray_frame, this->objects_pts[i], new_object_pts, status, err, 
                Size(31, 31), // window size, default
                21); // number of levels (-1), default
        // other parameters left as default

   
        // Check for errors and discard not found points
        for(int j=status.size()-1; j>=0; j--){
            // 0 means that LK doesen't find the point
            if (status[i] == 0) {
                new_object_pts.erase(new_object_pts.begin()+j);
                this->objects_pts[i].erase(this->objects_pts[i].begin()+j);
                err.erase(err.begin()+j);
            }
        }
        if (new_object_pts.size() == 0){ // too many points eliminated
            cout << "[ERROR]: tracking failed! No points detected, retune the parameters" << endl;
            this->objects_pts[i].clear();
            continue;
        }

        // Delete the higher errors
        double mean = accumulate(err.begin(), err.end(), 0.0) / err.size();
        double stdev = sqrt(inner_product(err.begin(), err.end(), err.begin(), 0.0) / err.size() - mean * mean);

        int mult;
        if      (err.size() >= 250) mult = 2; 
        else if (err.size() >= 200) mult = 3;
        else if (err.size() >= 150) mult = 4;
        else if (err.size() >= 100) mult = 5; // conservative

        for(int j=err.size()-1; j>=0; j--){
            if (err[j] > mean + mult*stdev) {
                new_object_pts.erase(new_object_pts.begin()+j);
                this->objects_pts[i].erase(this->objects_pts[i].begin()+j);
            }
        }

        if (new_object_pts.size() < 4){ // Homography needs at least some points
            cout << "[ERROR]: tracking failed! Few points detected, retune the parameters" << endl;
            this->objects_pts[i].clear();
            continue;
        }

        // Compute Homograpy to map the rectangle to new position    
        Mat H = findHomography(this->objects_pts[i], new_object_pts, 0);
        
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
             (std::chrono::high_resolution_clock::now() - start);
       
        if (INFO){
              cout << "[INFO]: tracking on object " << i << " computed in "
                  << duration.count()/1000.0 << " msec." << endl;
        }
        

        // Draw on output image
        if (draw_tracking){            
            for(Point2f pt: new_object_pts) circle(tracking_mask, pt, 1, colors[i]);
            add(output_img, tracking_mask, output_img);
        }
        if (draw_keypoints) 
            for(Point2f pt: new_object_pts) circle(output_img, pt, 3, colors[i], -1);
        drawRectangleFrame(output_img, H, colors[i], i);
        
        // Save new calculated point as for next step
        this->objects_pts[i] = new_object_pts;
    }
  
    // Switching to new frame for next computation
    this->prev_frame = next_frame.clone();
}


void Tracker::drawRectangle(Mat& img, Mat homography, Size obj_size, Scalar color, int i){
    int obj_cols = obj_size.width, obj_rows = obj_size.height;

    vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0,0);                //Top left
    obj_corners[1] = Point(obj_cols, 0);        //Top right
    obj_corners[2] = Point(obj_cols, obj_rows); //Bottom right
    obj_corners[3] = Point(0, obj_rows);        //Bottom left
    
    vector<Point2f> frame_corners(4);

    perspectiveTransform(obj_corners, frame_corners, homography);
    
    vector<Point> pts;
    for(int i=0;i<frame_corners.size();i++) pts.push_back(frame_corners.at(i));
    
    
    polylines(img, pts, true, color, 4);   
    this->obj_corners[i] = frame_corners;
}


void Tracker::drawRectangleFrame(Mat& img, Mat homography, Scalar color, int i){
    vector<Point2f> frame_corners(4);

    perspectiveTransform(this->obj_corners[i], frame_corners, homography);
    
    vector<Point> pts;
    for(int i=0;i<frame_corners.size();i++) pts.push_back(frame_corners.at(i));
    
    polylines(img, pts, true, color, 4);
    
    this->obj_corners[i] = frame_corners;
}

Size Tracker::adaptedSize(Mat img){
    int SCREEN_HEIGHT = 500, SCREEN_WIDTH = 1000;
    if (img.rows > SCREEN_HEIGHT || img.cols > SCREEN_WIDTH){
        
        if (SCREEN_HEIGHT - img.rows > SCREEN_WIDTH - img.cols)
            return Size(SCREEN_WIDTH, img.rows*SCREEN_WIDTH/img.cols);
        else
            return Size(img.cols*SCREEN_HEIGHT/img.rows, SCREEN_HEIGHT);
    }
    else return img.size();
}
