#ifndef GPUPROCESSOR_H
#define GPUPROCESSOR_H

#include "Processor.h"
class GpuProcessor : public Processor 
{
private:	
	int numberOfThreadsPerBlock;	
	
	int numberOfDevice;
	
	bool activated;
	
public:	
	GpuProcessor();
	
	void setNumberOfThreadsPerBlock(int numberOfThreadsPerBlock);
	
	int getNumberOfThreadsPerBlock();
	
	void setNumberOfDevice(int numberOfDevice);
	
	int getNumberOfDevice();
		
	Result* calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);	
	
	virtual Result* calculate(int numOfFeatures, 
		char** label0FeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
		char** label1FeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
		bool* featureMask);		
	
protected:

	virtual int getNumberOfFeatureSizeTimesSampleSize2dArrays(int numOfFeatures);		
	
};

#endif
