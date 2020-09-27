#include "../include/utils.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

Size adaptedSize(Mat img){
    if (img.rows > SCREEN_HEIGHT or img.cols > SCREEN_WIDTH){
        
        if (SCREEN_HEIGHT - img.rows > SCREEN_WIDTH - img.cols)
            return Size(SCREEN_WIDTH, img.rows*SCREEN_WIDTH/img.cols);
        else
            return Size(img.cols*SCREEN_HEIGHT/img.rows, SCREEN_HEIGHT);
    }
    else return img.size();
}

Size adaptedSize(Mat img, Size target){
    if (img.rows < img.cols) target = Size(target.height, target.width);

    if (img.rows > target.height or img.cols > target.width){
        if (target.height - img.rows > target.width - img.cols)
            return Size(target.width, img.rows*target.width/img.cols);
        else
            return Size(img.cols*target.height/img.rows, target.height);
    }
    else return img.size();
}


//void floydWarshall(vector<Superpixel> superpixels, Mat& distance_matrix){
    //int n = superpixels.size();
    //vector<vector<int>> D;
    //D.resize(n);
    //for(vector<int>& row: D) row.resize(n);

    //for(Superpixel sp: superpixels){
        //for(int neighbor: sp.neighbors){
            //cout << sp.id << " " << neighbor << " ~" << n << endl; 
            //cout << D[sp.id][neighbor] << endl;
            //D[sp.id][neighbor] = l1(sp.center, superpixels[neighbor].center);
        //}
    //}
    //for(Superpixel sp: superpixels) D[sp.id][sp.id] = 0;

    //for(Superpixel spk: superpixels){
        //for(Superpixel spi: superpixels){
            //for(Superpixel spj: superpixels){

                //D[spi.id][spj.id] = min(D[spi.id][spj.id], 
                                        //D[spi.id][spk.id] + D[spk.id][spj.id]);

            //}
        //}
    //}

    //distance_matrix = Mat(Size(n, n), CV_32S);
    //memcpy(distance_matrix.data, D.data(), n*n*sizeof(int));
//}

