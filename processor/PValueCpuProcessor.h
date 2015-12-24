#ifndef PVALUECPUPROCESSOR_H
#define PVALUECPUPROCESSOR_H

#include "CpuProcessor.h"
class PValueCpuProcessor : public CpuProcessor 
{
	private:
		double calculate_Pvalue(char *array1, int array1_size, char *array2, int array2_size);
	public:
		/*
        virtual Result* calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);
		*/
		virtual Result* fastCalculate(char** label0Array, char** label1Array, int label0Size, int label1Size, int numOfSamples, int numOfFeatures, bool* featureMask);

};

#endif
