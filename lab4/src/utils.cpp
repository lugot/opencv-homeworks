#include <algorithm>
#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "../include/utils.h"

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
    if (img.rows > img.cols)
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


void drawLine(cv::Mat& img, cv::Vec3f line, int size, cv::Scalar color){
    double rho = line[0], theta = line[1];
    
    cv::Point p1, p2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    
    int max_px = std::max(img.cols, img.rows);

    p1.x = cvRound(x0 + max_px*(-b));
    p1.y = cvRound(y0 + max_px*(a));
    p2.x = cvRound(x0 - max_px*(-b));
    p2.y = cvRound(y0 - max_px*(a));
     
    cv::line(img, p1, p2, color, size, cv::LINE_AA);
}

void drawCircle(cv::Mat& img, cv::Vec4f circle, int size, cv::Scalar color){
    cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
    int radius = cvRound(circle[2]);
    
    // draw the circle center
    cv::circle(img, center, 2, color, -1, 8, 0 );

    // draw the circle outline
    cv::circle(img, center, radius, color, size, 8, 0 );
}

void drawLinesOnImage(cv::Mat& img, std::vector<cv::Vec3f> lines){
    
    // color of lines: if only 2: red else blue
    cv::Scalar color = (lines.size()<=2 ? cv::Scalar(0,0,255)
                                        : cv::Scalar(255,0,0)); 

    // draw all the lines in red
    for(cv::Vec3f line: lines) drawLine(img, line, 1, color);

    // i know that is std::vector<cv::Vec3f>::const_iterator, be kind 
    auto max_line_iter = std::max_element(lines.begin(), lines.end(),
            [](const cv::Vec3f l1, const cv::Vec3f l2) -> bool {
                return l1[2] < l2[2];
            }); 
    
    if (max_line_iter != lines.end()) drawLine(img, *max_line_iter, 2, cv::Scalar(0,0,255));
    // this causes relocation (and inefficency)
    if (max_line_iter != lines.end()) lines.erase(max_line_iter);

    // second max
    max_line_iter = std::max_element(lines.begin(), lines.end(),
            [](const cv::Vec3f l1, const cv::Vec3f l2) -> bool {
                return l1[2] < l2[2];
        });

    if (max_line_iter != lines.end()) drawLine(img, *max_line_iter, 2, cv::Scalar(0,0,255)); 
}

void drawCirclesOnImage(cv::Mat& img, std::vector<cv::Vec4f> circles){
    
    // draw all the lines in blue
    for(cv::Vec4f circle: circles) drawCircle(img, circle, 2, cv::Scalar(255,0,0));

    if (circles.size() >= 1){
        cv::Vec4f circle = *std::max_element(circles.begin(), circles.end(),
            [](const cv::Vec4f c1, const cv::Vec4f c2) -> bool {
                return c1[3] < c2[3];
            });

        // green color for the most significant one 
        drawCircle(img, circle, 3, cv::Scalar(0,255,0));
    }
}
