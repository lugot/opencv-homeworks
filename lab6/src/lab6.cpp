#include <iostream>

#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/video/tracking.hpp>

#include "../include/global.h"
#include "../include/tracker.h"

using namespace std;
using namespace cv;
   
int INFO, DEBUG, SCREEN_WIDTH, SCREEN_HEIGHT;

int main(int argc, char** argv){
    
    const String keys = 
        "{help h usage ?     |              | print this message                                  }"
        "{@video_path        |data/video.mov| path to video                                       }"
        "{@objects_path      |data/covers   | path to set of images                               }"
        "{@objects_extension |jpg           | file extension of objects                           }"
        "{ratio r            |3.0           | ratio used to discard matches                       }"
        "{manual m           |0             | manually compute next frame by pressing any key     }"
        "{skip_frames s      |0             | number of frame skipped                             }"
        "{draw_keypoints d   |0             | 1 if you want to draw the keypoint                  }"
        "{draw_tracking t    |0             | 1 if you want to draw the movement of the keypoints }"
        "{verbosity v        |0             | verbosity, 1: INFO, 2: DEBUG                        }"
        "{height             |500           | screen height (for visualisation)                   }"
        "{width              |1000          | screen width (for visualisation)                    }"
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
    
    SCREEN_HEIGHT = parser.get<int>("height");
    SCREEN_WIDTH  = parser.get<int>("width");

    String video_path        = parser.get<String>("@video_path"), 
           objects_path      = parser.get<String>("@objects_path"),
           objects_extension = parser.get<String>("@objects_extension");
    double ratio             = max(1.0, parser.get<double>("ratio"));
    bool manual              = parser.get<bool>("manual");
    int skip_frames          = max(0,  parser.get<int>("skip_frames"));
    bool draw_keypoints      = parser.get<bool>("draw_keypoints");
    bool draw_tracking       = parser.get<bool>("draw_tracking") & draw_keypoints;

   
    if (!parser.check()) {
        parser.printErrors();
        return 0;
    }

    // Video acquiring
    VideoCapture cap;
    if (!cap.open("../" + video_path)){ //Check if the video is loaded correctly
        cout << "ERROR: no video found. Aborted" << endl;
        return 1;
    }

    // Images acquiring
    vector<String> filesname;
    try {
        utils::fs::glob("../" + objects_path, "*." + objects_extension, filesname);
    } catch (cv::Exception){ // no image found (directory not exist)
        cout << "No images found. Abort." << endl;
        return 1; 
    } 
    if (filesname.size() == 0){ // no images found (extension)
        cout << "No image found. Abort." << endl;
        return 1; 
    } 
    vector<Mat> objects;
    for(int i =0;i<filesname.size();i++){
        Mat object = imread(filesname[i]);
        objects.push_back(object);
    }

    
    // Tracking    
    Mat frame, next_frame, show_img; 
    cap >> frame;
    
    Tracker tr = Tracker(objects, ratio, draw_keypoints, draw_tracking);
    tr.matchFirstFrame(frame, show_img);
   
    namedWindow(GENERAL_WINDOW);
    int i=0;
    while(cap.isOpened()){
        i++;
        cap >> next_frame; 
        if(next_frame.empty()) break;        
 
        // Skip frames
        if (i%(skip_frames+1)) continue;

        auto start = std::chrono::high_resolution_clock::now();
       
        // Capture new frame and compute the tracking
        tr.trackFrame(next_frame, show_img);

        if (show_img.data == NULL){
            cout << "Tracking failed!" << endl;
            return 1;
        }

        auto duration = std::chrono::duration_cast<std::chrono::microseconds> 
             (std::chrono::high_resolution_clock::now() - start);
       
        if (INFO){
            double FPS = 900*1000.0/(duration.count()); 
            cout << "[INFO]: frame " << i << ": " << FPS << " FPS (" << duration.count()/1000.0 << " msec)" << endl << endl;
        }
        
        
        // Show result
        resize(show_img, show_img, Tracker::adaptedSize(show_img));
        imshow(GENERAL_WINDOW, show_img);
        if (manual) waitKey(0); // wait the user
        else        waitKey(1); // wait 1ms and procede to next frame
    }


    // When everything done, release the video capture object
    cap.release();
    // Closes all the frames
    destroyAllWindows();
    cout << "End of the video!" << endl;
    return 0;
}


