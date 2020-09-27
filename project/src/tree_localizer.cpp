#include "../include/tree_localizer.h"
#include "../include/superpixels_segmenter.h"
#include "../include/global.h"
#include "../include/utils.h"
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

TreeLocalizer::TreeLocalizer(TrainingParameters params) 
        :bow_descriptorextractor(new SiftDescriptorExtractor,
                             new BFMatcher){

    // Create BOW
    Ptr<BFMatcher> matcher = new BFMatcher;
    Ptr<SiftDescriptorExtractor> extractor = SIFT::create();
    bow_descriptorextractor = BOWImgDescriptorExtractor(extractor, matcher);

    // Create SIFT detector
    detector = SIFT::create();

    // Params
    this->dataset_path    = params.dataset_path;
    this->revocabulary    = params.revocabulary;
    this->vocabulary_size = params.vocabulary_size;
    this->num_clusters    = params.num_clusters;
    this->retrain         = params.retrain;
    this->training_size   = params.training_size;
    this->tree_color      = params.tree_color;
}

void TreeLocalizer::buildVocabulary(){
    
    Mat vocabulary_descriptors;

    if (!revocabulary){
        FileStorage fs("../data/vocabulary.yml", FileStorage::READ);
        if (INFO) cout << "[INFO]: Load vocabulary from file" << endl;
        
        if (fs.isOpened()){
            
            // Load vocabulary saved and..
            fs["vocabulary"] >> vocabulary_descriptors;
            // .. set it in the BOW extractor
            bow_descriptorextractor.setVocabulary(vocabulary_descriptors);
            
            fs.release();
            return;
        }
        else { // File not found..
            if (INFO) cout << "[INFO]: file not found! Need to rebuild vocabulary" << endl;
        }
    }

    if (INFO) cout << "[INFO]: building vocabulary" << endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    vector<String> filenames;
    try{
        utils::fs::glob("../" + dataset_path + "/rgb", "*.jpg", filenames);
    } 
    catch (cv::Exception){
        cout << "\n[ERROR] training set directory not found! Abort." << endl;
        exit(1);
    }
    random_shuffle(filenames.begin(), filenames.end());
    
    // Collecting keypoints and descriptors
    for(int i=0; i<vocabulary_size; i++){
        Mat img = imread(filenames[i]), descriptors;
        vector<KeyPoint> keypoints;

        // Step1: compute keypoints (globally)
        detector->detect(img, keypoints);
        // Computing descriptors
        detector->compute(img, keypoints, descriptors);


        // Step2: segment the image
        SuperpixelsSegmenter segmenter = SuperpixelsSegmenter(img);
        segmenter.segment();
        vector<Mat> aggregated_descriptors(segmenter.superpixels.size());

        // Step3: assing the keypoints to the relative superpixel
        for(int i=0; i<keypoints.size(); i++){
            int superpixel_id = segmenter.superpixels_map.at<int>(keypoints[i].pt);

            aggregated_descriptors[superpixel_id].push_back(descriptors.row(i));
        }

        // Step4: aggregate the descriptor: simply take the mean all over the 128-vector
        for (Mat &descriptor : aggregated_descriptors){
            Mat new_descriptors = Mat(1, descriptor.size().width, descriptor.type(), Scalar(0));

            // Compute the mean vector
            for (int i = 0; i < descriptor.size().width; i++) {
                int mean_col = static_cast<int>(mean(descriptor.col(i)).val[0]);

                new_descriptors.at<float>(0, i) = mean_col;
            }

            descriptor = new_descriptors.clone();
        }

        // Step5: add to vocabulary
        for(Mat descriptor: aggregated_descriptors){
            if (descriptor.size() == Size(0,0)) continue;

            // Add the aggregate descriptor as an entry in the vocabulary
            vocabulary_descriptors.push_back(descriptor);
        }

    }// end forall image
    
    if (INFO) cout << "[INFO]: vocabulary unclustered size " << vocabulary_descriptors.size() << endl;


    // Clustering
    BOWKMeansTrainer bowtrainer(num_clusters);
    vocabulary_descriptors = bowtrainer.cluster(vocabulary_descriptors);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
         (std::chrono::high_resolution_clock::now() - start);
    if (INFO) cout << "[INFO]: dictionary built in " << duration.count()/1000.0 << " msec." << endl;

    // Set the vocabulary in the BOW extactor
    bow_descriptorextractor.setVocabulary(vocabulary_descriptors);
    
    // Saving to file
    FileStorage fs("../data/vocabulary.yml", FileStorage::WRITE);
    fs << "vocabulary" << vocabulary_descriptors;
    fs.release();
}


void TreeLocalizer::prepareTrainingDataset(Ptr<ml::TrainData>& training_set, bool delete_duplicates, bool balance_set){

    // Fetch images
    vector<String> filenames_col, filenames_seg; 

    try{
        utils::fs::glob("../" + dataset_path + "/rgb", "*.jpg", filenames_col);
    } 
    catch (cv::Exception){
        cout << "\n[ERROR] training set directory not found! Abort." << endl;
        exit(1);
    }

    if (filenames_col.size() == 0){
        cout << "\n[ERROR] no training images found! Abort." << endl;
        exit(1);
    }

    // Randomize the training images
    random_shuffle(filenames_col.begin(), filenames_col.end());

    // Prepare the filenames for segmented images
    filenames_seg.reserve(filenames_col.size());
    int i,j;
    for(String s: filenames_col){

        i = s.length();
        while(filenames_col[0][i] != '/') i--;
        i++;

        j = s.length();
        while(filenames_col[0][j] != '.') j--;
        j--;
    
        String filename = "../" + dataset_path + "/seg/" + s.substr(i, j-i+1) + "_seg.png";
        filenames_seg.push_back(filename);
    }

    // Prepare containers
    Mat training_data = Mat(0, bow_descriptorextractor.descriptorSize(), CV_32F), labels;
    vector<int> superpixels_labels; 
    Mat null_histogram = Mat(Size(num_clusters, 1), CV_32F, Scalar(0.0));


    for(int i=0; i<training_size; i++){

        if (INFO) cout << "[INFO] training images:" << 
            "\n\t  colored image: " << filenames_col[i] <<
            "\n\tsegmented image: " << filenames_seg[i] << endl; 

        Mat img = imread(filenames_col[i]),
            seg = imread(filenames_seg[i]);
       
        SuperpixelsSegmenter segmenter = SuperpixelsSegmenter(img);
        segmenter.segment(200);
        segmenter.computeHistograms(detector, bow_descriptorextractor);
        //segmenter.aggregateHistograms(0); No aggregation
        segmenter.assignLabels(seg, tree_color, tree_ratio);

        // Build training set
        for(Superpixel& sp: segmenter.superpixels){

            if (sp.label == 0) continue;

            // Skips null histograms
            if (countNonZero(sp.histogram != null_histogram) == 0) continue;
            training_data.push_back(sp.histogram);
            superpixels_labels.push_back(sp.label);
        }

        if (INFO) cout << "[INFO] image " << i << " (" << filenames_col[i] <<  ") prepared for training" << endl;
    } 


    // Delete duplicates if requested, very computational demanding!
    if (delete_duplicates){

        if (INFO) cout << "[INFO]: deleting duplicates" << endl;

        Mat unique_training_data;
        vector<int> unique_superpixels_labels;
        for(int y=0; y<training_data.rows; y++){

            Mat row = training_data.row(y);

            // Search for other euqals row
            bool unique_row = true;
            for(int k=0; k<unique_training_data.rows; k++){
                if (countNonZero(unique_training_data.row(k) != row) == 0){ 
                    // found equal row
                    unique_row = false;
                    break;
                }
            }

            if (unique_row){
                unique_training_data.push_back(row);
                unique_superpixels_labels.push_back(superpixels_labels[y]);
            }
        }
        training_data = unique_training_data;
        superpixels_labels = unique_superpixels_labels;
    }

    int positive, negative;
    positive = negative = 0;

    // Balance the training set (positive vs negative response) if requested
    if (balance_set){
        for(int l: superpixels_labels){
            if (l == -1) negative++;
            else         positive++;
        }

        if (INFO) cout << "[INFO]: balancing training dataset" << endl;

        int diff = negative - positive;

        for(int y=training_data.rows-1; y>=0 && diff>0; y--){
            if (superpixels_labels[y] == -1){
                superpixels_labels.erase(y+superpixels_labels.begin());

                // Delete a row = move the memory up
                memmove(training_data.data +     y*training_data.cols, 
                        training_data.data + (y+1)*training_data.cols, 
                        training_data.cols * sizeof(float)*(training_data.rows-y-1));

                diff--;
            }
        }
        // Cut the matrix
        training_data = training_data.rowRange(0, superpixels_labels.size());
    }


    positive = negative = 0;
    for(int l: superpixels_labels){
        if (l == -1) negative++;
        else         positive++;
    }

    if (INFO) cout << "[INFO]: found " << positive << " positive (tree) data points and " << 
        negative << " negative (non-tree) data points"<< endl;

    // Convert labels to Mat for SVM
    Mat(superpixels_labels).copyTo(labels);


    // Prepare a TrainData object
    training_set = ml::TrainData::create(training_data, ml::ROW_SAMPLE, labels);
}

void TreeLocalizer::train(){

    // Check if it is possible to avoid training
    if (!retrain){
        if (INFO) cout << "[INFO]: Load SVM weigths from file" << endl;
        
        try { svm = ml::SVM::load("../data/svm.yml"); } 
        catch(const cv::Exception e) { // File not found..
            if (INFO) cout << "[INFO]: file not found! Need to retrain svm" << endl;
        }

        return;
    }


    // Prepare training set
    Ptr<ml::TrainData> training_set;
    prepareTrainingDataset(training_set, false, false);

    // Prepare SVM with parameters
    svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::CHI2);
    //svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 150, FLT_EPSILON));
    

    cout << "[INFO] start training .." << flush;
    auto start = std::chrono::high_resolution_clock::now();

    // Train and use k fold cross validation
    svm->trainAuto(training_set, 5);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
         (std::chrono::high_resolution_clock::now() - start);
    if (INFO) cout << " done in " << duration.count()/1000.0 << " msec." << endl;

    // Save the model to file
    svm->save("../data/svm.yml");

    if (INFO){
        vector<float> resp;
        float training_error = svm->calcError(training_set, false, resp);
        cout << "[INFO] training error: " << training_error << endl;
    }
}

void TreeLocalizer::localize_trees(cv::Mat img, cv::Mat& out, int num_superpixels, int N, bool draw_segmentation){
    
    if (INFO) cout << "[INFO] localizing.." << endl;

    // Segment the image
    SuperpixelsSegmenter segmenter = SuperpixelsSegmenter(img);
    segmenter.segment(num_superpixels);

    // Compute the histograms
    segmenter.computeHistograms(detector, bow_descriptorextractor);
    if (DEBUG){
        cout << "[DEBUG] responses without histogram aggregation" << endl;

        Mat debug_out;
        segmenter.predictLabels(svm);
        drawResponses(img, debug_out, segmenter.superpixels);

        resize(debug_out, debug_out, adaptedSize(debug_out));
        imshow(GENERAL_WINDOW, debug_out);
        waitKey(0);
    }

    // Aggregate
    segmenter.aggregateHistograms(N);
    if (DEBUG){
        cout << "[DEBUG] responses with histogram aggregation (N = " << N << ")" << endl;

        Mat debug_out;
        segmenter.predictLabels(svm);
        drawResponses(img, debug_out, segmenter.superpixels);

        resize(debug_out, debug_out, adaptedSize(debug_out));
        imshow(GENERAL_WINDOW, debug_out);
        waitKey(0);
    }

    // Predict the labels
    segmenter.predictLabels(svm, left_sigmoid_weight, right_sigmoid_weight);
    
    // Refine with a CRF
    //segmenter.refine(); ***

    //if (DEBUG){ ***
        //cout << "[DEBUG] responses with histogram aggregation (N = " << N << ") and CRF" << endl;

        //Mat debug_out;
        //segmenter.predictLabels(svm);
        //drawResponses(img, debug_out, segmenter.superpixels);

        //resize(debug_out, debug_out, adaptedSize(debug_out));
        //imshow(GENERAL_WINDOW, debug_out);
        //waitKey(0);
    //}


    // Merge and draw bouding boxes
    out = img.clone();

    vector<Superpixel> trees;
    segmenter.mergeSuperpixels(trees);

    for(Superpixel tree: trees){
        if (tree.label < 0.5) continue;
        Rect bounding_box = tree.getBoundingBox();

        rectangle(out, bounding_box, Scalar(0, 0, 255), 2);

        if (draw_segmentation){
            for(Point pt: tree.points) out.at<Vec3b>(pt) = Vec3b(0,255,0);
        }
    }
}

void TreeLocalizer::drawResponses(Mat img, Mat& out, vector<Superpixel> superpixels){

    out = img.clone();

    // Compute vector of resposes
    vector<float> predictions(superpixels.size());
    for(int i=0; i<predictions.size(); i++){
        predictions[i] = superpixels[i].label;
    }
    
    Mat color_gradient = Mat(1, 1000, CV_8UC3, Scalar(0.0));
    for(int x=0; x<1000; x++){
        if (x<500) color_gradient.at<Vec3b>(0,x) = Vec3b(0, 0, 255*x/1000);
        else       color_gradient.at<Vec3b>(0,x) = Vec3b(0, 255*x/1000, 0);
    }

    for(Superpixel sp: superpixels) for(Point p: sp.points){
        out.at<Vec3b>(p) = color_gradient.at<Vec3b>(0, sp.label*1000);
    }
}
