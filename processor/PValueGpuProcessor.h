#ifndef PVALUEGPUPROCESSOR_H
#define PVALUEGPUPROCESSOR_H

#include "GpuProcessor.h"
class PValueGpuProcessor : public GpuProcessor 
{
	public:
        	Result* calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);
			
			Result* calculate(int numOfFeatures, 
				char** label0ProcessingUnitFeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
				char** label1ProcessingUnitFeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
				bool* featureMask);

};

#endif
