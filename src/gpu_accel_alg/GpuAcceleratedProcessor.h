#ifndef GPUACCELERATEDPROCESSOR_H
#define GPUACCELERATEDPROCESSOR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "Processor.h"


class GpuAcceleratedProcessor : public Processor 
{
private:	
	int numberOfThreadsPerBlock;	
	
	int numberOfDevice;
			
public:	
	GpuAcceleratedProcessor();
	
	void setNumberOfThreadsPerBlock(int numberOfThreadsPerBlock);
	
	int getNumberOfThreadsPerBlock();
	
	void setNumberOfDevice(int numberOfDevice);
	
	int getNumberOfDevice();
			
	Result* calculate(int numOfFeatures, 
		char** label0FeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
		char** label1FeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
		int* numOfFeaturesPerArray,
		bool* featureMask);
		
	//for running in threadpool
	virtual void asynCalculateOnDevice(int maxFeaturesPerDevice,
		char* label0FeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
		char* label1FeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples,		
		bool* featureMask,		
		double* score,
		int device,
		cudaStream_t* stream){};
	
protected:

	int getNumberOfFeatureSizeTimesSampleSize2dArrays(int numOfFeatures);
		
};

#endif
