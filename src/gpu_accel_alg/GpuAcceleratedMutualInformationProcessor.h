#ifndef GPUACCELERATEDMUTUALINFORMATIONPROCESSOR_H
#define GPUACCELERATEDMUTUALINFORMATIONPROCESSOR_H

#include "GpuAcceleratedProcessor.h"
#include <string>

using namespace std;

class GpuAcceleratedMutualInformationProcessor : public GpuAcceleratedProcessor 
{
	public:
	
		GpuAcceleratedMutualInformationProcessor();
			
		void calculateOnDeviceWithStream(int** numberOfFeaturesPerStream,
			char*** label0SamplesArray_stream_feature, int numOfLabel0Samples,
			char*** label1SamplesArray_stream_feature, int numOfLabel1Samples,
			bool*** featureMasksArray_stream_feature,		
			double*** score,			
			cudaStream_t** streams,
			bool* success, string* errorMessage);


};

#endif
