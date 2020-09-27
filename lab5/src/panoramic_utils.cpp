#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "../include/global.h"
#include "../include/panoramic_utils.h"

cv::Mat PanoramicUtils::cylindricalProj(const cv::Mat& image, const double angle){
    
    cv::Mat tmp,result;
    cv::cvtColor(image, tmp, cv::COLOR_BGR2GRAY); //i think you used C API
    result = tmp.clone();

    double alpha(angle / 180 * CV_PI);
    double d((image.cols / 2.0) / tan(alpha));
    double r(d/cos(alpha));
    double d_by_r(d / r);
    int half_height_image(image.rows / 2);
    int half_width_image(image.cols / 2);

    for(int x = - half_width_image + 1, 
            x_end = half_width_image; x < x_end; ++x){
        
        for(int y = - half_height_image + 1,
                y_end = half_height_image; y < y_end; ++y){
        
            double x1(d * tan(x / r));
            double y1(y * d_by_r / cos(x / r));

            if (x1 < half_width_image &&
                x1 > - half_width_image + 1 &&
                y1 < half_height_image &&
                y1 > -half_height_image + 1){
          
                result.at<uchar>(y + half_height_image, x + half_width_image)
                    = tmp.at<uchar>(round(y1 + half_height_image),
                                    round(x1 + half_width_image));
            }//endif
        }//end innerfor
    }//end for

    return result;
}

void PanoramicUtils::imshowMultiple(cv::String winname, std::vector<cv::Mat> images){
    cv::namedWindow(winname);
    
    for(cv::Mat img: images){
        cv::imshow(winname, img);
        cv::waitKey();
    }
    
    cv::destroyAllWindows();
}
        
void PanoramicUtils::imshowMultipleKeypoints(cv::String winname, 
        std::vector<cv::Mat> images,
        std::vector<std::vector<cv::KeyPoint>> keypoints){

    cv::namedWindow(winname);

    for(int i=0; i<images.size(); i++){
        std::cout << "\timage " << i << ": extracted " << keypoints[i].size() << 
            " keypoints" << std::endl;

        cv::Mat res;
        cv::drawKeypoints(images[i], keypoints[i], res);
        
        cv::imshow(winname, res);
        cv::waitKey();
    }

    cv::destroyAllWindows();
}

void PanoramicUtils::imshowMultipleKeypointsMatched(cv::String winname, 
            std::vector<cv::Mat> images,
            std::vector<std::vector<cv::KeyPoint>> keypoints,
            std::vector<std::vector<cv::DMatch>> matches){
     
    cv::namedWindow(winname);
    for(int i=0; i<images.size()-1; i++){
        std::cout << "\tdrawing " << matches[i].size() << " matches between image " <<
            i << " and " << i+1 << "\n";

        cv::Mat res;
        cv::drawMatches(images[i], keypoints[i],
                images[i+1], keypoints[i+1],
                matches[i], res);

        cv::imshow(winname, res);
        cv::waitKey();
    }
    cv::destroyAllWindows();
    return ;
}


cv::Size PanoramicUtils::adaptedSize(cv::Mat img){
    if (img.rows > SCREEN_HEIGHT or img.cols > SCREEN_WIDTH){
        
        if (SCREEN_HEIGHT - img.rows > SCREEN_WIDTH - img.cols)
            return cv::Size(SCREEN_WIDTH, img.rows*SCREEN_WIDTH/img.cols);
        else
            return cv::Size(img.cols*SCREEN_HEIGHT/img.rows, SCREEN_HEIGHT);
    }
    else return img.size();
}

