#ifndef SUPERPIXEL_H
#define SUPERPIXEL_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

class Superpixel { 
public:
    /**
     * A superpixel is a region in an image with some spatial, color and
     * texture coherence. Default constructor.
     */
    Superpixel();
    /**
     * A superpixel is a region in an image with some spatial, color and
     * texture coherence. Constructor.
     *
     * @param id id of the superpixel
     * @param points position of the superpixel in the original image
     * @param center of the superpixel. Can be computed from points, keep
     * for convenience
     * @param color average color over the superpixel
     * @param label used for classification, both discrte and probabilistic
     * @param histogram used for classification
     * @param n_features number of features, linked to histogram. Can act as
     * a confidence
     */
    Superpixel(int id,
            std::vector<cv::Point> points, 
            cv::Point center, 
            cv::Vec3b color,
            float label, 
            cv::Mat histogram,
            int n_features
            );

    /**
     * Compute the l2 distance between HSV color of superpixel
     *
     * @param sp1 first superpixel
     * @param sp2 second superpixel
     * @return the distance as described above
     */
    static float HSV_distance(Superpixel sp1, Superpixel sp2);
    /**
     * Compute the l2 distance between HSV color of superpixel
     *
     * @param sp1 first superpixel
     * @param sp2 second superpixel
     * @return the distance as described above
     */
    static float LUV_distance(Superpixel sp1, Superpixel sp2);
    /**
     * Compute the l1 distance between superpixels, considering also the
     * color component.
     * The distance is weighted on the color component.
     *
     * @param sp1 first superpixel
     * @param sp2 second superpixel
     * @param weight a weight that acts as regularization on the color
     * component
     * @param size the size of the original image, used to 
     * @return the distance as described above
     */
    static int spatialcolor_distance(Superpixel sp1, Superpixel sp2, int weight);
    /**
     * Compute the l1 distance between two superpixels's centers
     *
     * @param sp1 first superpixel
     * @param sp2 second superpixel
     * @return the l1 distance bewtween centers
     */
    static int spatial_distance(Superpixel sp1, Superpixel sp2);

    /**
     * Merge this superpixel with another one, mataining the first supeixel
     * id. The process merges the points and the histogram. The center and 
     * the average color are recomputed.
     *
     * The functions suppose that all the fields are non-empty
     *
     * @param sp1 first superpixel
     * @param sp2 second superpixel
     * @param output
     */
    static void merge(Superpixel sp1, Superpixel sp2, Superpixel& merged);

    /**
     * Get the bouding box of the superpixel.
     *
     * @return the bounding box
     */
    cv::Rect getBoundingBox();
   


    /**
     * Fields
     * there are no getter&setters, everything is fully accessible
     */
    int id;
    std::vector<cv::Point> points;
    cv::Point center;
    float label;
    cv::Vec3b color;
    cv::Mat histogram;
    int n_features;
};

#endif // SUPERPIXEL_H
