#include "../include/global.h"
#include "../include/tree_localizer.h"
#include "../include/utils.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;
   
int INFO, DEBUG, SCREEN_WIDTH, SCREEN_HEIGHT;

int main(int argc, char** argv){

    srand(time(NULL));   
    
    const String keys = 
        "{help h usage ?    |                            | print this message                           }"
        "{@image_path       | data/test5.jpg             | path to image                                }"
        "{verbosity v       | 0                          | verbosity, 1: INFO, 2: DEBUG                 }"
        "{dataset_path      | data/training_set_selected | training set path (with rgb and seg subdirs  }"
        "{revocabulary rv   | 0                          | revocabulary                                 }" 
        "{vocabulary_size   | 100                        | number of image used to build the vocabulary }"
        "{num_clusters      | 400                        | number of clusters in vocabulary build       }"
        "{retrain rt        | 0                          | retrain                                      }" 
        "{training_size     | 150                        | number of image used to train the SVM        }"
        "{draw_segmentation | 0                          | draw the postive response                    }"
        "{height            | 600                        | screen height (for visualisation)            }"
        "{width             | 1000                       | screen width (for visualisation)             }"
        ;

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")){
        parser.printMessage();
        return 0;
    }
    
    switch(parser.get<int>("verbosity")){
        case 1:
            INFO = 1;
            DEBUG = 0;
            cout << "INFO mode: more infos will be printed\n"; 
            break;
        case 2:
            INFO = 1;
            DEBUG = 1;
            cout << "INFO mode: more infos will be printed\n"; 
            cout << "DEBUG mode: more infos will be printed\n";
            break;
        default:
            INFO = 0;
            DEBUG = 0;
            break;
    }

    setUseOptimized(true);
    setNumThreads(4);
    
    SCREEN_HEIGHT = parser.get<int>("height");
    SCREEN_WIDTH  = parser.get<int>("width");


    // Training process parameters
    String dataset_path = parser.get<String>("dataset_path"); 
    bool revocabulary   = parser.get<bool>("revocabulary"); 
    int vocabulary_size = parser.get<int>("vocabulary_size");
    int num_clusters    = parser.get<int>("num_clusters");
    bool retrain        = parser.get<bool>("retrain"); 
    int training_size   = parser.get<int>("training_size");
    // ADE20k is assumed for both tree color and files position
    Vec3b tree_color    = Vec3b(112, 32, 176);
    
    
    // Image path
    bool draw_segmentation = parser.get<bool>("draw_segmentation"); 
    String image_path      = parser.get<String>("@image_path"); 

    if (!parser.check()){
        parser.printErrors();
        return 0;
    }
   
    // Set training parameters
    TrainingParameters params = {dataset_path, 
        revocabulary, 
        vocabulary_size,
        num_clusters,
        retrain, 
        training_size,
        tree_color };

    // Create localizer object 
    TreeLocalizer tl = TreeLocalizer(params);
    tl.buildVocabulary();
    tl.train();

    Mat img = imread("../" + image_path);
    if (img.size() == Size(0,0)){
        cout << "\n[ERROR] image not found! Abort." << endl;
        exit(1);
    }

    Mat out;
    tl.localize_trees(img, out, 450, 1, draw_segmentation);

    namedWindow(GENERAL_WINDOW);
    resize(out, out, adaptedSize(out));
    imshow(GENERAL_WINDOW, out);
    waitKey(0);


    return 0;
}
// made with <3 by Dark-Powered Vim
// lugot

