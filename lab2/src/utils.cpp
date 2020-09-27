#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "../include/global.h"

void imshowHorizontalSided(std::string windowname, cv::Mat img, cv::Mat img2){
    // check on images compatibility
    if (img.type() != img2.type() or img.size() != img2.size())
        return;

    int width = SCREEN_WIDTH/2; //we have to side two images

    // defining target size in order that they fits the screen
    cv::Size target_size = cv::Size(width, img.rows*width/img.cols);

    resize( img,  img, target_size);
    resize(img2, img2, target_size);
    
    cv::Mat tiled = cv::Mat::zeros(img.rows, SCREEN_WIDTH, img.type());
    
    img.copyTo( tiled.rowRange(0, img.rows).colRange(    0,   width));
    img2.copyTo(tiled.rowRange(0, img.rows).colRange(width, 2*width));

    imshow(windowname, tiled);
}

void imshowVerticalSided(std::string windowname, cv::Mat img, cv::Mat img2){
    // check on images compatibility
    if (img.type() != img2.type() or img.size() != img2.size())
        return;
    
    int height = SCREEN_HEIGHT/2; //we have to stack two images
    
    // defining target size in order that they fits the screen
    cv::Size target_size = cv::Size(img.cols*height/img.rows, height);

    resize( img,  img, target_size); 
    resize(img2, img2, target_size); 
    
    cv::Mat tiled = cv::Mat::zeros(height*2, img.cols, img.type());
    
    img.copyTo( tiled.rowRange(       0,   img.rows).colRange(0, img.cols));
    img2.copyTo(tiled.rowRange(img.rows, 2*img.rows).colRange(0, img.cols));

    imshow(windowname, tiled);
}

void imshowSided(std::string windowname, cv::Mat img, cv::Mat img2){
    // selector of the best screen-fitting solution
    if (img.rows < img.cols)
        imshowHorizontalSided(windowname, img, img2);
    else
        imshowVerticalSided(windowname, img, img2); 
}

cv::Size adaptedSize(cv::Mat img){
    if (img.rows > SCREEN_HEIGHT or img.cols > SCREEN_WIDTH){
        
        if (SCREEN_HEIGHT - img.rows > SCREEN_WIDTH - img.cols)
            return cv::Size(SCREEN_WIDTH, img.rows*SCREEN_WIDTH/img.cols);
        else
            return cv::Size(img.cols*SCREEN_HEIGHT/img.rows, SCREEN_HEIGHT);
    }
    else return img.size();
}

void find3DCorners(int nrows, int ncols, std::vector<cv::Point3f> &corners, 
        double square_size){
    corners.resize(0);
    
    for(int i=0; i<nrows; i++) for(int j=0; j<ncols; j++)
        corners.push_back(cv::Point3f(i*square_size, j*square_size, 0.0));
}

double l2dist(cv::Point2f p1, cv::Point2f p2){
    return std::sqrt(std::pow(p1.x - p2.x, 2) +
                     std::pow(p1.y - p2.y, 2));
}
