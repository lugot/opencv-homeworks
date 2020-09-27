#include <opencv2/highgui.hpp>
#include "../include/global.h"
#include "../include/filter.h"

void onGaussianBlurKernelSize(int position, void* data){
	 GaussianFilter* gf = static_cast<GaussianFilter*>(data);

	 // do not refilter if the kernel size is the same
	 if (gf->getSize() != position){
		  // 0 is the minimun value a trackbar can hold
		  if (position == 0) position++;
		  gf->setSize(position);

		  // perform filtering..
		  gf->doFilter();
		  // .. and show result
		  imshow(GAUSSIAN_FILTER_WINDOW, gf->getResult());
	 }
}

void onGaussianBlurSigma(int position, void* data){
	 GaussianFilter* gf = static_cast<GaussianFilter*>(data);

	 // cannot skip due to double precision
	 gf->setSigma(position/10.0); //10.0 -> conversion (trackbar is int)
	 
	 // perform filtering..
	 gf->doFilter();
	 // .. and show result
	 imshow(GAUSSIAN_FILTER_WINDOW, gf->getResult());
}


void onMedianFilterKernelSize(int position, void* data){
	 MedianFilter* mf = static_cast<MedianFilter*>(data);
	
	 // do not refilter if the kernel size is the same
	 if (mf->getSize() != position){
		  // 0 is the minimun value a trackbar can hold
		  if (position == 0) position++;
		  mf->setSize(position);
		  
		  // perform filter..
		  mf->doFilter();
		  // .. and show result
		  imshow(MEDIAN_FILTER_WINDOW, mf->getResult());
	 }
}


void onBilateralFilterSigmaRange(int position, void* data){
	BilateralFilter* bf = static_cast<BilateralFilter*>(data);

	bf->setSigmaRange(position+0.0);
	bf->doFilter();

	 // perform filtering..
	 bf->doFilter();
	 // .. and show result
	 imshow(BILATERAL_FILTER_WINDOW, bf->getResult());
}

void onBilateralFilterSigmaSpace(int position, void* data){
	BilateralFilter* bf = static_cast<BilateralFilter*>(data);

	bf->setSigmaSpace(position+0.0);
	bf->doFilter();

	 // perform filtering..
	 bf->doFilter();
	 // .. and show result
	 imshow(BILATERAL_FILTER_WINDOW, bf->getResult());
}
