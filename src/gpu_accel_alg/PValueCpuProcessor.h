#ifndef PVALUECPUPROCESSOR_H
#define PVALUECPUPROCESSOR_H

#include "CpuProcessor.h"
class PValueCpuProcessor : public CpuProcessor 
{
	public:
		
		Result* calculate(int numOfFeatures, 
				char** label0FeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
				char** label1FeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
				bool* featureMask);

};

#endif
