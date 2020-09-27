#include "../include/superpixel.h"
#include <opencv4/opencv2/imgproc.hpp>
#include <climits>

using namespace std;
using namespace cv;

Superpixel::Superpixel(int id, vector<Point> points, Point center, Vec3b color, float label, Mat histogram, int n_features){
    this->id = id;
    this->points = points;
    this->center = center;
    this->color = color;
    this->label = label;
    this->histogram = histogram;
    this->n_features = n_features;
}

float Superpixel::HSV_distance(Superpixel sp1, Superpixel sp2){
    //l2 dist is used
    Mat3f color_i, color_j; 
    Mat3b(sp1.color).convertTo(color_i, CV_32FC3);
    Mat3b(sp1.color).convertTo(color_j, CV_32FC3);

    cvtColor(color_i, color_i, COLOR_BGR2HSV);
    cvtColor(color_j, color_j, COLOR_BGR2HSV);

    float dist = sqrt( pow(color_i[0] - color_j[0], 2) +
                                   pow(color_i[1] - color_j[1], 2) +
                                   pow(color_i[2] - color_j[2], 2));

    return dist;
}

float Superpixel::LUV_distance(Superpixel sp1, Superpixel sp2){
    //l2 dist is used
    Mat3f color_i, color_j; 
    Mat3b(sp1.color).convertTo(color_i, CV_32FC3);
    Mat3b(sp1.color).convertTo(color_j, CV_32FC3);

    cvtColor(color_i, color_i, COLOR_BGR2Luv);
    cvtColor(color_j, color_j, COLOR_BGR2Luv);

    float dist = sqrt( pow(color_i[0] - color_j[0], 2) +
                                   pow(color_i[1] - color_j[1], 2) +
                                   pow(color_i[2] - color_j[2], 2));

    return dist;
}

int Superpixel::spatialcolor_distance(Superpixel sp1, Superpixel sp2, int weight){
    // l1 weighted distance is used
    int dist = abs(sp1.center.x - sp2.center.x) + 
               abs(sp1.center.y - sp2.center.y) +
               abs(sp1.color[0] - sp2.color[0])*weight + 
               abs(sp1.color[1] - sp2.color[1])*weight + 
               abs(sp1.color[2] - sp2.color[2])*weight;

    return dist;
}

int Superpixel::spatial_distance(Superpixel sp1, Superpixel sp2){
    // l1 distance is used
    int dist = abs(sp1.center.x - sp2.center.x) + 
               abs(sp1.center.y - sp2.center.y);

    return dist;
}

void Superpixel::merge(Superpixel sp1, Superpixel sp2, Superpixel& merged){
    merged.id = sp1.id;

    merged.points.clear();
    merged.points.reserve(sp1.points.size() + sp2.points.size());
    merged.points.insert(merged.points.begin(), sp1.points.begin(), sp1.points.end());
    merged.points.insert(merged.points.begin(), sp2.points.begin(), sp2.points.end());

    merged.center = (sp1.center + sp2.center)/2;

    merged.color = (sp1.color + sp2.color)/2;

    merged.label = (sp1.label + sp2.label)/2.0;

    merged.histogram = (sp1.histogram + sp2.histogram);
    float l1 = 0; // normalization
    for(int x=0; x<merged.histogram.cols; x++) l1 += merged.histogram.at<float>(0, x);
    if (l1 != 0.0) merged.histogram = merged.histogram / l1;

    merged.n_features = (sp1.n_features + sp2.n_features)/2;
}

Rect Superpixel::getBoundingBox(){
    int max_x, max_y, min_x, min_y;
    max_x = max_y = 0;
    min_x = min_y = INT_MAX;
    
    // Compute max and min position of the superpixel
    for(Point p: points){
        max_x = max(max_x, p.x);
        max_y = max(max_y, p.y);
        min_x = min(min_x, p.x);
        min_y = min(min_y, p.y);
    }
    Point   top_left = Point(min_x, min_y), 
        bottom_right = Point(max_x, max_y);

    return Rect(top_left, bottom_right);
}
