#include <iostream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>

#include "../include/global.h"
#include "../include/panoramic_utils.h"
#include "../include/panoramic_image.h"

PanoramicImage::PanoramicImage(double ratio){ 
    setRatio(ratio);
}

int PanoramicImage::loadImages(cv::String directory, cv::String file_extension){

    // Loading filenames
    std::vector<cv::String> filenames;
    try {
        cv::utils::fs::glob("../" + directory, "*." + file_extension, filenames);
    } catch (cv::Exception){ return 1; } // no images founded (directory not exist)
    if (filenames.size() == 0){ return 1; } // no images founded (extension)

    // Loading images in a vector of Mat
    images.reserve(filenames.size());
    for(cv::String filename: filenames) images.push_back(cv::imread(filename)); 

    // setting right FOV depending on dataset
    if (directory.substr(directory.length()-6) == "dolomites") fov = 54.0;
    else                                                       fov = 66.0;

    if (INFO){
        std::cout << "INFO: loaded " << images.size() << " images from " << directory <<
            " directory. Set FOV = " << fov << std::endl;
    }

    return 0; //ok
}

void PanoramicImage::project(){
    
   auto start = std::chrono::high_resolution_clock::now();
   
   // perform projection
    for(cv::Mat& img: images) img = PanoramicUtils::cylindricalProj(img, fov/2);

   auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
         (std::chrono::high_resolution_clock::now() - start);
   
   if (INFO){
      std::cout << "INFO: performed projection on " << images.size() <<
            " images in " << duration.count()/1000.0 << " msec." << std::endl;
   }
    if (DEBUG){
        std::cout << "DEBUG: projected images" << std::endl;
        PanoramicUtils::imshowMultiple(GENERAL_WINDOW, images);
        std::cout << "\n";
    }
}

int PanoramicImage::extractFeatures(){
   auto start = std::chrono::high_resolution_clock::now();
    
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
   
   // perform features extraction 
    sift->detect(images, keypoints); //image set way
    sift->compute(images, keypoints, descriptors);
    
   auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
         (std::chrono::high_resolution_clock::now() - start);
   
   if (INFO){
      std::cout << "INFO: performed SIFT detection on " << images.size() <<
            " images in " << duration.count()/1000.0 << " msec." << std::endl;
    }
    if (DEBUG){
        std::cout << "\nDEBUG: images with features" << std::endl;
        PanoramicUtils::imshowMultipleKeypoints(GENERAL_WINDOW, images, keypoints);
        std::cout << "\n";
    }
    
    // Panoramic computation can fail if theres not enought keypoints
    for(std::vector<cv::KeyPoint> set_keypoints: keypoints){
        if (set_keypoints.size() == 0){
            return 1; // not enough keypoints
        }
    }

    return 0;
}

void PanoramicImage::matchKeypoints(){
   auto start = std::chrono::high_resolution_clock::now();
    
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, true);
   
   // perform matching 
    matches.resize(keypoints.size()-1);
    int i=0; 
    for(std::vector<cv::DMatch>& match: matches){
        matcher->match(descriptors[i], descriptors[i+1], match);
        i++;
    }

   auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
         (std::chrono::high_resolution_clock::now() - start);
   
   if (INFO){
      std::cout << "INFO: performed brute force matching on " << images.size() <<
            " images (in pairs) in " << duration.count()/1000.0 << " msec." << std::endl;
    }
    if (DEBUG){
        std::cout << "\nDEBUG: matches" << std::endl;
        PanoramicUtils::imshowMultipleKeypointsMatched(GENERAL_WINDOW, images, keypoints, matches);
        std::cout << "\n";
    }
}

int PanoramicImage::refineMatches(){
    
    // PART 1: matches elimination based on ratio 
    
    // reserve a vector containing the reduced set of matches
    std::vector<std::vector<cv::DMatch>> reduced_matches;
    reduced_matches.reserve(matches.size());
   
    // iterate over the sets of matches
    for(std::vector<cv::DMatch> set_matches: matches){
        std::vector<cv::DMatch> new_matches;
    
        float min_distance = set_matches[0].distance;
        for(cv::DMatch match: set_matches) // mindistance between pairs of keypoints
            min_distance = std::min(min_distance, match.distance);
    
        // keep match if undet min*ratio
        for(cv::DMatch match: set_matches)
            if (match.distance < min_distance*ratio) new_matches.push_back(match);
      
       
        if (INFO){
            std::cout << "INFO: reduced matches: " << set_matches.size() <<
                " -> " << new_matches.size() << std::endl;
        }
        
        reduced_matches.push_back(new_matches);
    }
    
    // new matches
    matches = reduced_matches;

    
    // PART 2: outliers eliminations
    
    std::vector<cv::Mat> masks;
    masks.resize(images.size()-1);
    int m=0; // index over masks

    for(int i=0; i<matches.size(); i++){
    
        if (matches[i].size() < 5){
            std::cout << "WARNING: on match " << i << 
                " few matches! Cannot exclude outliers" << std::endl;
            continue;
        }

        // build vectors of Point2f
        std::vector<cv::Point2f> src_points, dst_points;
        for(cv::DMatch match: matches[i]){
            src_points.push_back(keypoints[ i ][match.queryIdx].pt);
            dst_points.push_back(keypoints[i+1][match.trainIdx].pt);
        }
        
        // compute mask, ignored H (returned from function)
        cv::findHomography(src_points, dst_points, cv::RANSAC, 3, masks[m++]);
        
        int num_deleted = 0;
        for(int j=masks[i].rows-1; j>=0; j--){
            if (masks[i].at<uchar>(j,0) == 0){ // outlier
                num_deleted++;
                matches[i].erase(matches[i].begin()+j);
            }
        }

        if (INFO){
            std::cout << "INFO: deleted " << num_deleted << " outliers" << std::endl;
        }
    }
    
    if (DEBUG){
        std::cout << "\nDEBUG: reduced matches" << std::endl;
        PanoramicUtils::imshowMultipleKeypointsMatched(GENERAL_WINDOW, images, keypoints, matches);
        std::cout << "\n";
    }
    
    // Panoramic computation can fail if theres not enought matches
    for(std::vector<cv::DMatch> set_matches: matches){
        if (set_matches.size() == 0){
            std::cout << "ERROR: not enought matching keypoints. Aborted." << 
                "Retune the parameters.";
            return 1;
        }
    }

    return 0;
}

void PanoramicImage::computeTranslations(){
    // reserve a vector of traslation 
    //  a tralsation is only a pair of point, the mean on image left and right 
    translations.reserve(matches.size());
    
    // iterate over matches
    for(int i=0; i<matches.size(); i++){
   
            
        // compute mean point on left and right image
        cv::Point2f left, right;
        for(cv::DMatch match: matches[i]){
            left  += keypoints[ i ][match.queryIdx].pt;
            right += keypoints[i+1][match.trainIdx].pt;
        }
        left  /= static_cast<int>(matches[i].size());
        right /= static_cast<int>(matches[i].size());
        
        // add traslation
        translations.push_back({left, right});

        if (INFO){
            std::cout << "INFO: Match " << i << ", translations: left " << left <<
                ", right " << right << std::endl;
        }
    }
}

void PanoramicImage::mergeImages(){
    // reserve a vector of images that will be merged to obtain the panoramic
    //  this is computed in two steps:
    //      1. the images are cropped on x axis, keeping the unmatched parted
    //         untouched
    //      2. then they are adjusted on the y axis using a affine trasformation
    //         matrix

    std::vector<cv::Mat> tiled(images.size());
    
    for(int i=0; i<tiled.size(); i++){
        if (i==0){ // first image
            cv::Point2f left;
            left = translations[i].first;
           
            // crop the image and keep only the left side, just before the feature
            tiled[i] = images[i].colRange(0, cvRound(left.x));
                
        } else if (i == tiled.size()-1){ // last image
            cv::Point2f right;
            right = translations[i-1].second;
            
            // crop the image and keep only the right side, just after the
            //  feature 
            tiled[i] = images[i].colRange(right.x, images[i].cols);
        } else {
            cv::Point2f left, right;
            right = translations[i-1].second;
            left  = translations[ i ].first;
   
            // crop the image and keep the central part, this iff there's no
            //  overlapping ie the user is very slow to move the camera (problem
            //  can be avoided with different the photo sampling due to 
            //  accelerometer, the fix is there)
            if (right.x < left.x)
                tiled[i] = images[i].colRange(right.x, left.x); 
        }
    }

    // deleting in case of overlapping (very rare actually)
    std::vector<cv::Mat>::const_iterator it;
    for(it = tiled.begin(); it != tiled.end(); it++){
        if (it->size().width == 0){
            tiled.erase(it);
            it--;
        }
    }

    // compute final result size (cropped later)
    int total_width = 0;
    for(cv::Mat img: tiled) total_width += img.cols;
    result = cv::Mat(images[0].rows, total_width, images[0].type());
   
    // y tralsations. float because I do subs
    float total_trasl = 0, pos_trasl = 0, neg_trasl = 0;
    for(int i=1; i<tiled.size(); i++){
        float y_left  = translations[i-1].first.y,
               y_right = translations[i-1].second.y;
    
        // calculate if there is y traslation between center of features 
        float actual_trasl = y_right - y_left;
        total_trasl += actual_trasl; // the tralsation follows the previous ones

        // set this value for final cropping
        if (actual_trasl > 0) // right image is higher than left
            pos_trasl = std::max(pos_trasl, actual_trasl);
        else //(actual_trasl <= 0) // right image is lower than left
            neg_trasl = std::max(neg_trasl, std::abs(actual_trasl));

        // to shift (up or down) an affine transformation is used
        std::vector<float> mat_data = {1, 0, 0, 0, 1, -actual_trasl};
        cv::Mat transformation_matrix(2, 3, CV_32F, mat_data.data());
        cv::warpAffine(images[i], images[i], transformation_matrix, images[0].size());
    }
   
    // merge all the cropped images
    int col_idx = 0;
    for(cv::Mat img: tiled){
        img.copyTo(result.colRange(col_idx, col_idx + img.cols));
        col_idx += img.cols;
    }

    // crop over y axis due to y alignment
    result = result.rowRange(neg_trasl, images[0].rows - pos_trasl);
}

cv::Mat PanoramicImage::elaborate(){
    
    // pipeline
    project();
    if (extractFeatures()){
        std::cout << "ERROR: features extraction failed! " << 
            "Too few keypoints, retune the parameter" << std::endl;

        result.data = NULL;
        return result;
    }
    matchKeypoints();
    if (refineMatches()){
        std::cout << "ERROR: matches refinition failed! " <<
            "Too few matches, retune the parameter" << std::endl;

        result.data = NULL;
        return result;
    }
    computeTranslations();
    mergeImages();
    cv::equalizeHist(result, result);

    return result;
}

cv::Mat PanoramicImage::getResult(){
    return result;
}

double PanoramicImage::getRatio(){ return ratio; }
void PanoramicImage::setRatio(double ratio){ this->ratio = ratio; }
