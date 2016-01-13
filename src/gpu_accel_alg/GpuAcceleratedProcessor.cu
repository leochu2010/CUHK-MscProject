#include "GpuAcceleratedProcessor.h"
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "threadpool/ThreadPool.h"
#include "utils/Timer.h"

using namespace std;

GpuAcceleratedProcessor::GpuAcceleratedProcessor(){
	this->numberOfThreadsPerBlock = 0;
	this->numberOfDevice = 0;
}

void GpuAcceleratedProcessor::setNumberOfThreadsPerBlock(int numberOfThreadsPerBlock)
{
		this->numberOfThreadsPerBlock = numberOfThreadsPerBlock;
}

int GpuAcceleratedProcessor::getNumberOfThreadsPerBlock()
{
		if (this->numberOfThreadsPerBlock >0){
			return numberOfThreadsPerBlock;
		}else{
			return 1024;
		}
}

void GpuAcceleratedProcessor::setNumberOfDevice(int numberOfDevice){
	this->numberOfDevice = numberOfDevice;
}

int GpuAcceleratedProcessor::getNumberOfDevice(){
	
	if (this->numberOfDevice >0){
		return numberOfDevice;
	}else{
		int deviceCount = 0;
		cudaGetDeviceCount(&deviceCount);
		return deviceCount;
	}
}

int GpuAcceleratedProcessor::getNumberOfFeatureSizeTimesSampleSize2dArrays(int numOfFeatures){
	return this->getNumberOfDevice();
}

struct CreateStreamArgs{
	cudaStream_t* stream;
	cudaError_t* streamResult;
	int dev;
};

//create stream
void createStream(void* arg) {	
	CreateStreamArgs* createStreamArgs = (CreateStreamArgs*) arg;
	
	cudaStream_t* stream = createStreamArgs->stream;
	cudaError_t* streamResult = createStreamArgs->streamResult;
	int dev = createStreamArgs->dev;
	
	cudaSetDevice(dev);
	*streamResult = cudaStreamCreate(stream);
	
	cout<<"created stream for device:"<<dev<<endl;
}

struct AsynCalculateArgs{
	int featuresPerDevice;
	char* label0FeatureSizeTimesSampleSize2dArray;
	int numOfLabel0Samples;
	char* label1FeatureSizeTimesSampleSize2dArray;
	int numOfLabel1Samples;
	int* numOfFeaturesPerArray;
	bool* featureMask;
	double* score;
	int device;
	cudaStream_t* stream;
	GpuAcceleratedProcessor* processor;	
};

void asynCalculate(void* arg){
	AsynCalculateArgs* calculateArgs = (AsynCalculateArgs*) arg;	
	
	calculateArgs->processor->asynCalculateOnDevice(		
		calculateArgs->featuresPerDevice,
		calculateArgs->label0FeatureSizeTimesSampleSize2dArray,
		calculateArgs->numOfLabel0Samples,
		calculateArgs->label1FeatureSizeTimesSampleSize2dArray,
		calculateArgs->numOfLabel1Samples,		
		calculateArgs->featureMask,
		calculateArgs->score,
		calculateArgs->device,
		calculateArgs->stream
	);
}

Result* GpuAcceleratedProcessor::calculate(int numOfFeatures, 
		char** label0ProcessingUnitFeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
		char** label1ProcessingUnitFeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
		int* numOfFeaturesPerArray,
		bool* featureMask){
		
	/*
	Step 1:
		for each device
			new thread for create stream
	
	Step 2:
		create results array
	
	Step 3:
		wait for all threads join 
	
	Step 4:
		for each device	
			new thread for asyn calculation
				pass data, devId, stream for asyn calculate
				pass resultArray for passing result back
	
	Step 5:
		wait for all threads join
	
	Step 6:
		return results;
	*/
	
	Timer totalProcessing("Total Processing");
	totalProcessing.start();
	
	//get device and thread numbers	
	int deviceCount = getNumberOfDevice();	
	//ThreadPool tp(deviceCount);
	ThreadPool tp(deviceCount);
	
	int ret = tp.initialize_threadpool();
	if (ret == -1) {
		cerr << "Failed to initialize thread pool!" << endl;
		exit(EXIT_FAILURE);
	}
	
	cudaStream_t stream[deviceCount];
	cudaError_t streamResult[deviceCount];
		
	for(int dev=0; dev<deviceCount; dev++) {
					
		CreateStreamArgs* createStreamArgs = new CreateStreamArgs;		
		createStreamArgs->stream = &stream[dev];
		createStreamArgs->streamResult = &streamResult[dev];
		createStreamArgs->dev = dev;
		
		Task* t = new Task(&createStream, (void*) createStreamArgs);
		tp.add_task(t);
	}
	
	//do sth else when waiting...
	//get feature num 
	int maxFeaturesPerDevice = getFeaturesPerArray(numOfFeatures, deviceCount);
	
	double **score = (double**)malloc(deviceCount * sizeof(double*));	
	for(int dev=0; dev<deviceCount; dev++) {
		score[dev] = (double*)malloc(numOfFeaturesPerArray[dev] * sizeof(double));		
	}
		
	cout<<"wait for steams ready"<<endl;
	tp.waitAll();
	
	Timer processing("Processing Time");
	//note that creating stream first can save some time, can further investigate if needed
	processing.start();
	
	for(int dev=0; dev<deviceCount; dev++) {		
		AsynCalculateArgs* calculateArgs = new AsynCalculateArgs;		
		calculateArgs->featuresPerDevice = numOfFeaturesPerArray[dev];
		calculateArgs->label0FeatureSizeTimesSampleSize2dArray = label0ProcessingUnitFeatureSizeTimesSampleSize2dArray[dev];
		calculateArgs->numOfLabel0Samples = numOfLabel0Samples;
		calculateArgs->label1FeatureSizeTimesSampleSize2dArray = label1ProcessingUnitFeatureSizeTimesSampleSize2dArray[dev];
		calculateArgs->numOfLabel1Samples = numOfLabel1Samples;
		calculateArgs->score = score[dev];
		calculateArgs->device = dev;
		calculateArgs->stream = &stream[dev];
		calculateArgs->processor = this;
		
		//cout << streamResult[dev] <<endl;
				
		Task* t = new Task(&asynCalculate, (void*) calculateArgs);
		tp.add_task(t);
	}		
	tp.waitAll();
	tp.destroy_threadpool();
	
	Result* calResult = new Result;		
	calResult->scores = new double[numOfFeatures];
		
	for(int i=0; i<numOfFeatures;i++){
		int dev = i / maxFeaturesPerDevice;
		int featureIdx = i % maxFeaturesPerDevice;
		calResult->scores[i] = score[dev][featureIdx];
		//cout<<dev<<","<<featureIdx<<","<<i<<":"<<score[dev][featureIdx]<<endl;
	}	
		
	for(int dev=0; dev<deviceCount; dev++) {
		free(score[dev]);
	}
	free(score);
	
	processing.stop();
	totalProcessing.stop();	
	calResult->startTime=processing.getStartTime();
	calResult->endTime=processing.getStopTime();	
	calResult->success=true;
	return calResult;	
			
	/*
		don't forget try stream features in warp size in child classes
	*/
			
	
	
}

