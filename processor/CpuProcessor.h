#ifndef CPUPROCESSOR_H
#define CPUPROCESSOR_H

#include "Processor.h"
class CpuProcessor : public Processor 
{
	private:
		int numberOfThreads;
	public:
		void setNumberOfThreads(int numberOfThreads);
		
		Result* calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);
		
		virtual Result* fastCalculate(char** label0Array, char** label1Array, int label0Size, int label1Size, int numOfSamples, int numOfFeatures, bool* featureMask);
};

#endif
