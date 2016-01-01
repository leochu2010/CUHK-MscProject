#ifndef PVALUECPUPROCESSOR_H
#define PVALUECPUPROCESSOR_H

#include "CpuProcessor.h"
class PValueCpuProcessor : public CpuProcessor 
{
	private:
		double calculate_Pvalue(char *array1, int array1_size, char *array2, int array2_size);
	public:
		
		Result* calculate(int numOfFeatures, 
				char** label0FeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
				char** label1FeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
				bool* featureMask);

};

#endif
