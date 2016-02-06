#ifndef GPUACCELERATEDTTESTPROCESSOR_H
#define GPUACCELERATEDTTESTPROCESSOR_H

#include "GpuAcceleratedProcessor.h"
#include <string>

using namespace std;

class GpuAcceleratedTTestProcessor : public GpuAcceleratedProcessor 
{
	public:
	
		GpuAcceleratedTTestProcessor();
	
		void calculateOnStream(int* numberOfFeaturesPerStream,
			char** label0SamplesArray_stream_feature, int numOfLabel0Samples,
			char** label1SamplesArray_stream_feature, int numOfLabel1Samples,
			bool** featureMasksArray_stream_feature,		
			double** score,
			int device,
			cudaStream_t* streams,
			bool* success, string* errorMessage);

};

#endif
