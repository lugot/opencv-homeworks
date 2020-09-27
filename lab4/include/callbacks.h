// This header file contain five callback declarations about the filters
//  and theirs parameters

void onEdgesDetectorMinThreshold(int position, void* userdata);
void onEdgesDetectorMaxThreshold(int position, void* userdata);

void onLinesDetectorRho(int position, void* userdata);
void onLinesDetectorTheta(int position, void* userdata);
void onLinesDetectorThreshold(int position, void* userdata);

void onCirclesDetectorDp(int position, void* userdata);
void onCirclesDetectorMinDist(int position, void* userdata);
void onCirclesDetectorCannyThreshold(int position, void *userdata);
void onCirclesDetectorAccumulatorThreshold(int position, void *userdata);

