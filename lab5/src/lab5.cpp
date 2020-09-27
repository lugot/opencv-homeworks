#include <vector>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "../include/global.h" // sorry again for this but my linter gets mad
#include "../include/panoramic_image.h"
#include "../include/panoramic_utils.h"

using namespace std;
using namespace cv;

// defining global flag
int DEBUG;
int INFO;
int SCREEN_HEIGHT;
int SCREEN_WIDTH;
    
int main(int argc, char** argv){
   
    const String keys = 
        "{help h usage ? |         | print this message                }"
        "{@path          |data/data| path to series of images          }"
        "{@extension     |bmp      | file extension of images               }"
        "{@ratio         |5.0      | ratio used to discard matches     }"
        "{verbosity v    |0        | verbosity, 1: INFO, 2: DEBUG      }"
        "{height         |500      | screen height (for visualisation) }"
        "{width          |1000     | screen width (for visualisation)  }"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
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
    
    SCREEN_HEIGHT = parser.get<int>("height");
    SCREEN_WIDTH  = parser.get<int>("width");

    String path           = parser.get<String>("@path"), 
           file_extension = parser.get<String>("@extension");
    double ratio          = parser.get<double>("@ratio");
    
    PanoramicImage pi = PanoramicImage(ratio);
    if (pi.loadImages(path, file_extension)){
        cout << "No image founded. Abort." << std::endl;
        return 1;
    }

    cv::Mat panoramic = pi.elaborate();
    resize(panoramic, panoramic, PanoramicUtils::adaptedSize(panoramic));
    
    cout << "Result: \n";
    namedWindow(GENERAL_WINDOW);
    imshow(GENERAL_WINDOW, panoramic);
    waitKey();
    

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
// STL methods, functions and containers -> snake_case
// OpenCV & custom functions -> camelCase
// variables -> snake_case
