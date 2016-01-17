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
	
	int numberOfStreamsPerDevice;
			
public:	
	GpuAcceleratedProcessor();
	
	void setNumberOfThreadsPerBlock(int numberOfThreadsPerBlock);
			
	void setNumberOfDevice(int numberOfDevice);	
	
	void setNumberOfStreamsPerDevice(int numberOfStreams);
			
	Result* calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);
	
	//for running in threadpool
	virtual void calculateOnStream(int* numberOfFeaturesPerStream,
		char** label0SamplesArray_stream_feature, int numOfLabel0Samples,
		char** label1SamplesArray_stream_feature, int numOfLabel1Samples,
		bool* featureMask,		
		double** score,
		int device,
		cudaStream_t* streams){};		
		
protected:

	int getNumberOfDevice();	
	
	int getNumberOfThreadsPerBlock();
	
	int getNumberOfStreamsPerDevice();
	
	Result* calculateOnDevice(int numOfFeatures, 
		char*** label0Samples_device_stream_feature, int numOfLabel0Samples,
		char*** label1Samples_device_stream_feature, int numOfLabel1Samples, 
		int** numberOfFeaturesPerStream,
		bool* featureMask);
		
};

#endif
