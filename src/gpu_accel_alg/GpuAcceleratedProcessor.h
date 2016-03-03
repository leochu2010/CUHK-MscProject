#ifndef GPUACCELERATEDPROCESSOR_H
#define GPUACCELERATEDPROCESSOR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "Processor.h"
#include <string.h>

using namespace std;

class GpuAcceleratedProcessor : public Processor 
{
private:
	int numberOfThreadsPerBlock;
	
	int numberOfDevice;
	
	int numberOfStreamsPerDevice;
	
	bool threadPoolEnabled;
			
public:	
	GpuAcceleratedProcessor();
	
	void setNumberOfThreadsPerBlock(int numberOfThreadsPerBlock);
			
	void setNumberOfDevice(int numberOfDevice);	
	
	void setNumberOfStreamsPerDevice(int numberOfStreams);
	
	void enableThreadPool();
			
	Result* calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);
	
	//for running in threadpool
	virtual void calculateOnStream(int* numberOfFeaturesPerStream,
		char** label0SamplesArray_stream_feature, int numOfLabel0Samples,
		char** label1SamplesArray_stream_feature, int numOfLabel1Samples,
		bool** featureMasksArray_stream_feature,		
		double** score,
		int device,
		cudaStream_t* streams,
		bool* success, string* errorMessage
		){};		
		
protected:

	int getNumberOfDevice();	
	
	int getNumberOfThreadsPerBlock();
	
	int getNumberOfStreamsPerDevice();
	
	Result* calculateOnDevice(int numOfFeatures, 
		char*** label0Samples_device_stream_feature, int numOfLabel0Samples,
		char*** label1Samples_device_stream_feature, int numOfLabel1Samples, 
		int** numberOfFeaturesPerStream,
		bool*** featureMasksArray_device_stream_feature,		
		bool* successPerDevice, string* errorMessagePerDevice);
	
	virtual void calculateAllFeatures(
			int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels,
			double* scores, bool* success, string* errorMessage){};
			
	Result* parallelizeCalculationOnFeatures(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);
		
	virtual Result* parallelizeCalculationOnStages(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, int* packedSampleFeatureMatrix, bool* featureMask, char* labels){
		return NULL;
	};
};

#endif
