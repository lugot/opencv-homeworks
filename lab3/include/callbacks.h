// This header file contain five callback declarations about the filters
//  and theirs parameters

void onGaussianBlurKernelSize(int position, void* userdata);
void onGaussianBlurSigma(int position, void* userdata);
void onMedianFilterKernelSize(int position, void* userdata);
void onBilateralFilterSigmaRange(int position, void* userdata);
void onBilateralFilterSigmaSpace(int position, void* userdata);
