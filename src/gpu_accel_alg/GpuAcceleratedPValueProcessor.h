#ifndef GPUACCELERATEDPVALUEGPUPROCESSOR_H
#define GPUACCELERATEDPVALUEGPUPROCESSOR_H

#include "GpuAcceleratedProcessor.h"
class GpuAcceleratedPValueProcessor : public GpuAcceleratedProcessor 
{
	public:

		void asynCalculateOnDevice(int maxFeaturesPerDevice,
			char* label0FeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
			char* label1FeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples,			
			bool* featureMask,		
			double* score,
			int device,
			cudaStream_t* stream);

};

#endif
