#ifndef SUPERPIXELS_SEGMENTER_H
#define SUPERPIXELS_SEGMENTER_H

#include "../include/superpixel.h"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include <vector>

class SuperpixelsSegmenter{
public:
    /**
     * Default constructor
     */
    SuperpixelsSegmenter(cv::Mat img);

    /**
     * Segment the image using SEEDS superpixels, compute centers, points and
     * connectivity. BGR is used with color. Classification fields (label, 
     * histogram and n_features) are set to default values.
     *
     * @param max_superpixels upper bound to the number of superpixels
     */
    void segment(int max_superpixels=200);

    /**
     * Compute an histogram for each superpixel given a SIFT  features detector
     * and an pre-trained BOW extractor.
     *
     * @param detector SIFT feature detector
     * @param extractor Bag Of Word exractor, with vocabulary already
     * learned.
     */
    void computeHistograms(cv::Ptr<cv::SIFT> detector, cv::BOWImgDescriptorExtractor extractor);

    /**
     * Aggregate the histogram of every superpixel with the ones of the
     * N-closest superpixels. A l1 normalization is performed.
     *
     * @param N the number of closest superpixels used in the histograms merging
     */
    void aggregateHistograms(int N);

    /**
     * Assign a label to each superpixel given a segmentation map in input. The
     * label is assigned if at least the a tunable percentage of region is covered with that
     * label. The possible labels are TREE and NON-TREE, an input color is
     * determinant to choose the correct label.
     *
     * @param segmentation_map Mat with same type of the input image with
     * colored region for labels
     * @param tree_color the BGR color of the TREE label in segmentation_map
     * @param tree_ratio minimum ratio between labels to consider a region as
     * tree
     */
    void assignLabels(cv::Mat segmentation_map, cv::Vec3b tree_color, float tree_ratio=0.95);

    /**
     * Predict and assaign the labels of the superpixel using a SVM. The result is
     * the probability, modulated by a sigmoid, to be part of the TREE class.
     *
     * @param svm the SVM used for the predictions
     * @param left_sigmoid_weight weight applied to negative response. Higher
     * value means that the resposnes will be even lower
     * @param right_sigmoid_weight weight applied to positive responses. Higher
     * value means that the resposnes will be even higher 
     */
    void predictLabels(cv::Ptr<cv::ml::SVM> svm, float left_sigmoid_weight=1, float right_sigmoid_weight=1);

    /**
     * Refine the predicted labels using a CRF.
     *
     * @param confidence_tradeoof a tradeoff between spatial consistency and the
     * confidence over the superpixel given by SVM. Higher value means that the
     * spatiality is less considered
     */
    //void refine(float confidence_tradeoff=1); ***

    /**
     * Merge the superpixels based on connected components created by labels.
     *
     * @param merged_superpixels the merged superpixels
     */
    void mergeSuperpixels(std::vector<Superpixel>& merged_superpixels);


    /**
     * Fields
     */
    cv::Mat img;
    std::vector<Superpixel> superpixels;
    cv::Mat superpixels_map;
    cv::Mat contour_mask;
    std::vector<std::vector<int>> adjacency_matrix;
};

#endif // SUPERPIXELS_SEGMENTER_H
