#ifndef UTILS_H
#define UTILS_H

// This header contains userful wrapper of imshow
#include "../include/global.h"
#include "../include/superpixel.h"
#include <opencv2/core.hpp>

/**
 * Compute the best Size to adapt the image dimension to your monitor, for
 * visualisation. No resize is performed
 *
 * @param img the image to resize
 * @return the adapted size to your screen
 */
cv::Size adaptedSize(cv::Mat img);

/**
 * Compute the best Size to adapt the image dimension to a specific monotor
 * dimension, for
 * visualisation. No resize is performed
 *
 * @param img the image to resize
 * @return the adapted size to your screen
 */
cv::Size adaptedSize(cv::Mat img, cv::Size target);

/**
 * Unused
 */
//void floydWarshall(std::vector<Superpixel> superpixels, cv::Mat& distance_matrix);

#endif // UTILS_H
