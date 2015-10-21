#ifndef PVALUECPUPROCESSOR_H
#define PVALUECPUPROCESSOR_H

#include "CpuProcessor.h"
class PValueCpuProcessor : public CpuProcessor 
{
	private:
			double calculate_Pvalue(int *array1, int array1_size, int *array2, int array2_size);
	public:
        	virtual Result* calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);

};

#endif
