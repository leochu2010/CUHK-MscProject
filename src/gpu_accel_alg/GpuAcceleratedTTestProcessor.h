#ifndef GPUACCELERATEDTTESTPROCESSOR_H
#define GPUACCELERATEDTTESTPROCESSOR_H

#include "GpuAcceleratedProcessor.h"
class GpuAcceleratedTTestProcessor : public GpuAcceleratedProcessor 
{
	public:
	
		void calculateOnStream(int* numberOfFeaturesPerStream,
			char** label0SamplesArray_stream_feature, int numOfLabel0Samples,
			char** label1SamplesArray_stream_feature, int numOfLabel1Samples,
			bool** featureMasksArray_stream_feature,		
			double** score,
			int device,
			cudaStream_t* streams);

};

#endif
