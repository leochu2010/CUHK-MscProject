#ifndef PVALUEGPUPROCESSOR_H
#define PVALUEGPUPROCESSOR_H

#include "GpuProcessor.h"
class PValueGpuProcessor : public GpuProcessor 
{
	public:
        	virtual Result* calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);

};

#endif
