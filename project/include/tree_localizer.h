#ifndef TREE_LOCALIZER
#define TREE_LOCALIZER

#include "superpixel.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>
#include <vector>

struct TrainingParameters {
    cv::String dataset_path;
    bool revocabulary;
    int vocabulary_size;
    int num_clusters;
    bool retrain;
    int training_size;
    cv::Vec3b tree_color;
};

class TreeLocalizer {
public:
    TreeLocalizer(TrainingParameters params);

    /**
     * Build the vocabulary. If the vocablary file is not found or it is
     * indicated in the constructor, the vocabulary is rebuild. 
     */
    void buildVocabulary();

    /**
     * Prepare a training set using the info from the costructor.
     *
     * @training_set training set to return
     * @param delete_duplicates delete the duplicates entries in the training
     * set
     * @param balance_set balance the false/positive ratio
     */
    void prepareTrainingDataset(cv::Ptr<cv::ml::TrainData>& training_set,
            bool delete_duplicates = false,
            bool balance_set = false);

    /**
     * Train an SVM model, CHI2 kernel and binary classification. If the v
     * model file is not found or it is indicated in the constructor, 
     * the model is retrained.
     */
    void train();

    /**
     * Localize and draw bouding boxes for trees.
     *
     * @param img the input image
     * @param out the output image
     * @param num_superpixel the number of superpixel to find, more is less
     * coarse 
     * @param N aggregation parameter 
     * @param draw_segmentation draw an addiatioanl segmentation map
     */
    void localize_trees(cv::Mat img, cv::Mat& out, int num_superpixels=250, int N=1, bool draw_segmentation=false);

    /**
     * Draw the probabilistic responses given an image and a segmentation.
     *
     * @param img the input image
     * @param out the output image
     * @param superpixels superpixel with responses
     */
    static void drawResponses(cv::Mat img, cv::Mat& out,
            std::vector<Superpixel> superpixels);

private:
    cv::Ptr<cv::SiftDescriptorExtractor> detector;
    cv::BOWImgDescriptorExtractor bow_descriptorextractor;
    cv::Ptr<cv::ml::SVM> svm;

    /**
     * Vocabulary and training parameters
     */
    cv::String dataset_path;
    bool revocabulary;
    int vocabulary_size;
    int num_clusters;
    bool retrain;
    int training_size;
    cv::Vec3b tree_color;

    /**
     * Parameters
     */
    float tree_ratio = 0.95;
    int N = 1;
    float  left_sigmoid_weight = 3;
    float right_sigmoid_weight = 2;

};


#endif // TREE_LOCALIZER
