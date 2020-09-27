#include "../include/global.h"
#include "../include/superpixels_segmenter.h"
#include "../include/utils.h"
#include <opencv2/core.hpp>
#include <opencv2/ximgproc/seeds.hpp>
//#include <DGM.h> ***
#include <types.h>
#include <iostream>
#include <vector>
#include <queue>

using namespace std;
using namespace cv;
using namespace DirectGraphicalModels;

SuperpixelsSegmenter::SuperpixelsSegmenter(Mat img){
    this->img = img.clone();
}

void SuperpixelsSegmenter::segment(int max_superpixels){

    if (INFO) cout << "[INFO] starting superpixel segmentation" << endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Preprocessing of the image
    Mat lab_img = img.clone();
    
    // Preparing image for superpixel segmentation
    GaussianBlur(lab_img, lab_img, Size(3,3), 1.0);
    cvtColor(lab_img, lab_img, COLOR_BGR2Lab);

    // Create SEEDS superpixels segmenter. An upper bound to the number of regions is fixed by parameter
    //  some other parameters are set to defult or picked with validation
    Ptr<ximgproc::SuperpixelSEEDS> SEEDS_segmenter = ximgproc::createSuperpixelSEEDS(
            img.cols, img.rows, img.channels(), 
            max_superpixels, 5, 5, 5, true);

    // Execute the segmentation. The number of iterations is high but still
    //  feasible
    SEEDS_segmenter->iterate(lab_img, 30);
    
    // Compute the mask and the map of superpixels
    SEEDS_segmenter->getLabelContourMask(contour_mask, true);
    SEEDS_segmenter->getLabels(superpixels_map);
    int num_superpixels = SEEDS_segmenter->getNumberOfSuperpixels();


    auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
         (std::chrono::high_resolution_clock::now() - start);
    if (INFO) cout << "[INFO] superpixel segmentation: detected " << num_superpixels << " superpixels " << 
        "in " << duration.count()/1000.0 << " msec." << endl;


    // Step1: prepare superpixels vector with default values
    superpixels.reserve(num_superpixels);
    for(int i=0; i<num_superpixels; i++){
        superpixels.push_back(Superpixel(i,                  // id 
                                        vector<Point>(),     // points
                                        Point(0, 0),         // center
                                        Vec3b(0,0,0),        // color
                                        0.0,                 // label: 0 means unassigned
                                        Mat(),               // histogram
                                        0                    // number of features
                                        ));
    }


    // Step2: fill with points
    for(int y=0; y<superpixels_map.rows; y++) for(int x=0; x<superpixels_map.cols; x++){
        // fill the points field
        superpixels[superpixels_map.at<int>(y,x)].points.push_back(Point(x,y));
        // fill also the center field so we avoid and +O(N) computation
        superpixels[superpixels_map.at<int>(y,x)].center += Point(x,y);
    }
    // then adjust centers
    for(Superpixel sp: superpixels){
        if (sp.points.size() == 0) continue;
        sp.center /= static_cast<int>(sp.points.size());
    }


    // Step3: compute neighbors
    adjacency_matrix.resize(num_superpixels);
    for(vector<int>& row: adjacency_matrix) row.resize(num_superpixels);

    for(Superpixel& sp: superpixels){
        // Find a neighbor in every direction
        Point dirs[] = {Point( 0,+1),
                        Point(+1, 0),
                        Point( 0,-1),
                        Point(-1, 0)};

        for(Point p: sp.points) for(Point dir: dirs){
            if ((p+dir).x < 0 ||
                (p+dir).x > lab_img.cols-1 ||
                (p+dir).y < 0 ||
                (p+dir).y > lab_img.rows-1) continue;
   
            // Populate adjacency matrix
            if (superpixels_map.at<int>(p+dir) != sp.id){
                adjacency_matrix[sp.id][superpixels_map.at<int>(p+dir)]++;
                adjacency_matrix[superpixels_map.at<int>(p+dir)][sp.id]++;
            }
        }
    }


    // Step4: compute average color (BGR)
    for(Superpixel& sp: superpixels){
        for(Point pt: sp.points) sp.color += img.at<Vec3b>(pt);

        // Sometimes an empty superpixel is tracked with some combination of superpixel algorithm  and parameters, need to check
        if (sp.points.size() != 0){
            sp.color[0] /= sp.points.size();
            sp.color[1] /= sp.points.size();
            sp.color[2] /= sp.points.size();
        }
    }


    if (DEBUG){ 
        cout << "[DEBUG] superpixel segmentation map" << endl;
        Mat mask, out = img.clone();

        mask = contour_mask.clone();
        cvtColor(mask, mask, COLOR_GRAY2BGR);

        add(mask, out, out);

        resize(out, out, adaptedSize(out));
        namedWindow(GENERAL_WINDOW);
        imshow(GENERAL_WINDOW, out);
        waitKey();
    }
}


void SuperpixelsSegmenter::computeHistograms(Ptr<SIFT> detector, BOWImgDescriptorExtractor extractor){

    if (INFO) cout << "[INFO] extracting features from superpixels" << endl;
    
    // Compute histogram for each superpixel
    for(Superpixel& sp: superpixels){

        if (sp.points.size() == 0){ // no points -> empty descriptor
            sp.histogram = Mat(Size(extractor.descriptorSize(), 1), CV_32F, Scalar(0.0));
            sp.n_features = 0;
            continue;
        }

        // Window surrounding the region
        Rect bounding_box = sp.getBoundingBox();
        Mat superpixel_window = img(bounding_box);


        // Detect keypoints
        vector<KeyPoint> keypoints;
        detector->detect(superpixel_window, keypoints);

        // Discard if the feature is in another superpixel (and not in this one)
        for(int i=keypoints.size()-1; i>=0; i--){

            // Keypoint position wrt original img
            Point keypoint_img = Point2f(keypoints[i].pt.x + bounding_box.x, 
                                         keypoints[i].pt.y + bounding_box.y); 
        
            if (superpixels_map.at<int>(keypoint_img) != sp.id){
                keypoints.erase(keypoints.begin()+i);
            }
        }

        if (DEBUG) cout << "[DEBUG] extracted " << keypoints.size() << " keypoints from superpixel " << sp.id << endl;     

        // Compute the histogram
        if (keypoints.size() < 2){ 
            // Suppress if there's not enough keypoints 
            sp.histogram = Mat(Size(extractor.descriptorSize(), 1), CV_32F, Scalar(0.0)); // 0 matrix
            sp.n_features = 0;
        }else{
            extractor.compute(superpixel_window, keypoints, sp.histogram);
            sp.n_features = keypoints.size(); 
        }

    }//end forall superpixel
}


void SuperpixelsSegmenter::aggregateHistograms(int N){

    if (INFO) cout << "[INFO] aggregating histograms (N=" << N <<  ")" << endl;
  
    // Useless but skips some computation
    if (N == 0) return;

    // Need to swap to avoid change something that it's running
    vector<Superpixel> new_superpixels(superpixels);
    int j=0;

    for(Superpixel& sp: superpixels){

        // To speed up computation, it is considered only a 2-level BFS starting from the superpixel
        vector<int> neighbors;
        for(int i=0; i<adjacency_matrix.size(); i++) if(adjacency_matrix[sp.id][i] > 0) neighbors.push_back(i);
        sort(neighbors.begin(), neighbors.end());

        if (DEBUG){
            cout << "[DEBUG]         neighbors of " << sp.id << ": ";
            for(int neighbor: neighbors) cout << neighbor << " ";
            cout << endl;
        }


        // Expand the neighborhood
        vector<int> expanded_neighbors;
        for(int neighbor: neighbors) for(int i=0; i<adjacency_matrix.size(); i++){
            if (adjacency_matrix[neighbor][i] > 0) expanded_neighbors.push_back(i);
        }
        sort(expanded_neighbors.begin(), expanded_neighbors.end());
        expanded_neighbors.erase( unique(expanded_neighbors.begin(), expanded_neighbors.end() ), expanded_neighbors.end() );

        if (DEBUG){
            cout << "[DEBUG] epanded neighbors of " << sp.id << ": ";
            for(int neighbor: expanded_neighbors) cout << neighbor << " ";
            cout << endl;
        }


        // Sort the neighborhood based on distance
        vector<pair<int, int>> topN_closer;
        for (int neighbor : expanded_neighbors) {
          topN_closer.push_back({Superpixel::HSV_distance(sp, superpixels[neighbor]), neighbor});
        }
        sort(topN_closer.begin(), topN_closer.end());

        // Add the N closest superpixels's histogram to current one
        for(int i=0; i<min(N, static_cast<int>(topN_closer.size())); i++){
            add(new_superpixels[j].histogram, superpixels[topN_closer[i].second].histogram, new_superpixels[j].histogram);
        }
        // l1 normalize the fresh new histogram 
        float l1 = 0;
        for(int x=0; x<new_superpixels[j].histogram.cols; x++) l1 += new_superpixels[j].histogram.at<float>(0, x);
        if (l1 != 0.0) new_superpixels[j].histogram = new_superpixels[j].histogram / l1;

        j++;
    }

    // Copy the superpixels with aggregated histograms into vector
    superpixels = new_superpixels;
}

void SuperpixelsSegmenter::assignLabels(Mat segmentation_map, Vec3b tree_color, float tree_ratio){

    if (INFO) cout << "[INFO] labels assignment from pre-segmented image" << endl;

    int superpixels_positive = 0;
    for(Superpixel& sp: superpixels){

        // Determine if the superpixel belong to a tree or not
        int count_positive = 0;
        for(Point pt: sp.points){
            if (segmentation_map.at<Vec3b>(pt) == tree_color) count_positive++;
        }
        
        // Class is TREE if some % of the region is covered
        if (count_positive > sp.points.size()*tree_ratio){
            superpixels_positive++;
            sp.label =  1;   //  1: TREE
        }else sp.label = -1; // -1: NON-TREE
    }

    if (INFO) cout << "[INFO] found " << superpixels_positive << " positive and " << 
        superpixels.size()-superpixels_positive << " negative superpixels"  << endl;
}

void SuperpixelsSegmenter::predictLabels(Ptr<ml::SVM> svm, float left_sigmoid_weight, float right_sigmoid_weight){

    if (INFO) cout << "[INFO] predicting labels" << endl;
    auto start = std::chrono::high_resolution_clock::now();

    for(Superpixel& sp: superpixels){
        // Predict: sigmoid is used to convert to probability
        float decision = svm->predict(sp.histogram, noArray(), ml::StatModel::RAW_OUTPUT);
        
        // Weight differently the sigmoid
        if (decision < 0) sp.label = 1/(1+exp( left_sigmoid_weight*decision));
        else              sp.label = 1/(1+exp(right_sigmoid_weight*decision));
    } 
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
         (std::chrono::high_resolution_clock::now() - start);

    if (INFO) cout << "[INFO] predicted " << superpixels.size() << " labels " << 
        "in " << duration.count()/1000.0 << " msec." << endl;
}

/*
 *  void SuperpixelsSegmenter::refine(float confidence_tradeoff){ ***
 *
 *    if (INFO) cout << "[INFO] labels refinement" << endl;
 *
 *    // Graph and decoder
 *    CGraphPairwise graph(2);
 *    CInferViterbi decoder(graph);
 *
 *    //Create nodes
 *    for(int i=0; i<superpixels.size(); i++) graph.addNode();
 *    // Create edges 
 *    for(int i=0; i<adjacency_matrix.size(); i++) for(int j=i+1; j<adjacency_matrix.size(); j++){
 *        if (adjacency_matrix[i][j] != 0) graph.addArc(i,j);
 *    }
 *    
 *
 *    // Node potential: probabilities from SVM
 *    Mat node_pot(2, 1, CV_32FC1);
 *    for(Superpixel sp: superpixels){
 *        node_pot.at<float>(0,0) =   sp.label;
 *        node_pot.at<float>(1,0) = 1-sp.label;
 *
 *        graph.setNode(sp.id, confidence_tradeoff*node_pot);
 *    }
 *
 *    // Edge potential: sort of default
 *    Mat edge_pot(2, 2, CV_32FC1);
 *    for(int i=0; i<adjacency_matrix.size(); i++) for(int j=i+1; j<adjacency_matrix.size(); j++){
 *        if (adjacency_matrix[i][j] > 0){
 *
 *            // Compute the LUV color difference (l2)
 *            float color_difference = Superpixel::LUV_distance(superpixels[i], superpixels[j]);
 *
 *            // Compute the potential
 *            float potential = adjacency_matrix[i][j] / (1 + color_difference);
 *            
 *            if (DEBUG){
 *                cout << "[DEBUG] potential between superpixel " << i << " and " << j << ": " << potential << endl;
 *            }
 *            
 *            // Set potential matrix
 *            edge_pot.at<float>(0, 0) = potential;
 *            edge_pot.at<float>(0, 1) = 0.0f;
 *            edge_pot.at<float>(1, 0) = 0.0f;
 *            edge_pot.at<float>(1, 1) = potential;
 *
 *            graph.setArc(i,j, edge_pot);
 *        }
 *    }
 *    
 *    cout << "[INFO] start decoding.." << flush;
 *    auto start = std::chrono::high_resolution_clock::now();
 *    // Solve the net
 *    decoder.decode(10000);
 *
 *    auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
 *         (std::chrono::high_resolution_clock::now() - start);
 *    if (INFO) cout << " done in " << duration.count()/1000.0 << " msec." << endl;
 *
 *
 *    // Get new computed potentials
 *    Mat potentials;
 *    graph.getNodes(0, superpixels.size(), potentials);
 *
 *    // Reassign labels
 *    for(int y=0; y<potentials.rows; y++) superpixels[y].label = potentials.at<float>(y,1);
 *}
 */

void SuperpixelsSegmenter::mergeSuperpixels(vector<Superpixel>& merged_superpixels){

    vector<int> visited(superpixels.size(), 0);
    vector<vector<int>> connected_components;
    queue<int> q;
    
    // Super easy BFS
    int num_visited = 0;
    q.push(0);
    visited[0] = 1;
    while(num_visited < superpixels.size()){

        vector<int> component;
        while(!q.empty()){
            int act = q.front(); q.pop();
            num_visited++;
            component.push_back(act);

            for(int i=0; i<adjacency_matrix.size(); i++){
                if (adjacency_matrix[act][i] == 0 || visited[i]) continue;
                
                if ((-1 + 2*superpixels[act].label) * (-1 + 2*superpixels[i].label) > 0){
                    visited[i] = 1;

                    q.push(i);
                }

            }
        }
        sort(component.begin(), component.end());
        connected_components.push_back(component);

        int i = 0;
        while(visited[i] != 0) i++;
        if (i < superpixels.size()){
            visited[i] = 1;

            q.push(i);
        }

    }
        
    if (INFO) cout << "[INFO] detected " << connected_components.size() << " connected components" << endl;

    if (DEBUG){
        for(vector<int> component: connected_components){
            cout << "[DEBUG] component: ";
            for(int supepixel: component) cout << supepixel << " ";
            cout << endl;
        }
    }

    if (DEBUG){
        cout << "[DEBUG] merging superpixels" << endl;
    }
    // For each cc merge the superpixels
    merged_superpixels.clear();
    int id = 0;
    for(vector<int> component: connected_components){
        if (component.size() <= superpixels.size()/250+1 || superpixels[component[0]].label < 0.5) continue;

        Superpixel merged = superpixels[component[0]];
        for(int i=1; i<component.size(); i++){
            Superpixel::merge(merged, superpixels[component[i]], merged);
        }
        merged.id = id++;

        merged_superpixels.push_back(merged);
    }
}
