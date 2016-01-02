#ifndef GPUACCELERATEDPVALUEGPUPROCESSOR_H
#define GPUACCELERATEDPVALUEGPUPROCESSOR_H

#include "GpuAcceleratedProcessor.h"
class GpuAcceleratedPValueProcessor : public GpuAcceleratedProcessor 
{
	public:
        	//Result* calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);
			
			Result* calculate(int numOfFeatures, 
				char** label0FeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
				char** label1FeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
				bool* featureMask);

};

#endif
