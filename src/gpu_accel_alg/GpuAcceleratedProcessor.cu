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
	this->numberOfStreamsPerDevice = 1;
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

void GpuAcceleratedProcessor::setNumberOfStreamsPerDevice(int numberOfStreamsPerDevice)
{
	this->numberOfStreamsPerDevice = numberOfStreamsPerDevice;
}

int GpuAcceleratedProcessor::getNumberOfStreamsPerDevice()
{
	return this->numberOfStreamsPerDevice;
}

struct CreateStreamArgs{
	cudaStream_t* stream;
	cudaError_t* streamResult;
	int numberOfStreamsPerDevice;
	int dev;
};

//create stream
void createStream(void* arg) {	
	CreateStreamArgs* createStreamArgs = (CreateStreamArgs*) arg;
	
	cudaStream_t* stream = createStreamArgs->stream;
	cudaError_t* streamResult = createStreamArgs->streamResult;
	int dev = createStreamArgs->dev;
	int numberOfStreamsPerDevice = createStreamArgs->numberOfStreamsPerDevice;
	
	cudaSetDevice(dev);
	for(int i=0; i<numberOfStreamsPerDevice; i++){
		streamResult[i] = cudaStreamCreate(&stream[i]);
	}
	
	cout<<"created stream for device:"<<dev<<endl;
}

struct AsynCalculateArgs{
	int* numberOfFeaturesPerStream;
	char** label0SamplesArray_stream_feature;
	int numOfLabel0Samples;
	char** label1SamplesArray_stream_feature;
	int numOfLabel1Samples;	
	bool** featureMasksArray_stream_feature;
	double** score;
	int device;
	cudaStream_t* stream;
	GpuAcceleratedProcessor* processor;	
};

void asynCalculate(void* arg){
	AsynCalculateArgs* calculateArgs = (AsynCalculateArgs*) arg;	
	
	calculateArgs->processor->calculateOnStream(
		calculateArgs->numberOfFeaturesPerStream,
		calculateArgs->label0SamplesArray_stream_feature,
		calculateArgs->numOfLabel0Samples,
		calculateArgs->label1SamplesArray_stream_feature,
		calculateArgs->numOfLabel1Samples,		
		calculateArgs->featureMasksArray_stream_feature,
		calculateArgs->score,
		calculateArgs->device,
		calculateArgs->stream
	);
}


Result* GpuAcceleratedProcessor::calculateOnDevice(int numOfFeatures, 
	char*** label0SamplesArray_device_stream_feature, int numOfLabel0Samples,
	char*** label1SamplesArray_device_stream_feature, int numOfLabel1Samples, 
	int** numberOfFeaturesPerStream,
	bool*** featureMasksArray_device_stream_feature){
		
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
	
	int streamCount = getNumberOfStreamsPerDevice();
	cudaStream_t stream[deviceCount][streamCount];
	cudaError_t streamResult[deviceCount][streamCount];
		
	for(int dev=0; dev<deviceCount; dev++) {
		
		CreateStreamArgs* createStreamArgs = new CreateStreamArgs;		
		createStreamArgs->stream = stream[dev];
		createStreamArgs->streamResult = streamResult[dev];
		createStreamArgs->dev = dev;
		createStreamArgs->numberOfStreamsPerDevice = getNumberOfStreamsPerDevice();
		
		Task* t = new Task(&createStream, (void*) createStreamArgs);
		tp.add_task(t);
	}
	
	//do sth else when waiting...
	//get feature num 
	int numberOfDevices = getNumberOfDevice();
	int featuresPerDevice = getFeaturesPerArray(numOfFeatures, numberOfDevices);
	int numberOfStreams = getNumberOfStreamsPerDevice();
	int featuresPerStream = getFeaturesPerArray(featuresPerDevice, numberOfStreams);
	
	double ***score = (double***)malloc(numberOfDevices * sizeof(double**));
	for(int i=0; i<numberOfDevices; i++) {
		score[i] = (double**)malloc(numberOfStreams * sizeof(double*));		
		for(int j=0; j<numberOfStreams; j++){
			score[i][j] = (double*)malloc(featuresPerStream * sizeof(double));		
		}
	}
		
	cout<<"wait for steams ready"<<endl;
	tp.waitAll();
	
	Timer processing("Processing Time");
	//note that creating stream first can save some time, can further investigate if needed
	processing.start();
	
	for(int dev=0; dev<deviceCount; dev++) {
		//cout << "[GpuAcceleratedProcessor]label0SamplesArray_device_stream_feature["<<dev<<"][0][0]=" << 0+label0SamplesArray_device_stream_feature[dev][0][0] << endl;
		AsynCalculateArgs* calculateArgs = new AsynCalculateArgs;
		calculateArgs->numberOfFeaturesPerStream = numberOfFeaturesPerStream[dev];
		calculateArgs->label0SamplesArray_stream_feature = label0SamplesArray_device_stream_feature[dev];
		calculateArgs->numOfLabel0Samples = numOfLabel0Samples;
		calculateArgs->label1SamplesArray_stream_feature = label1SamplesArray_device_stream_feature[dev];
		calculateArgs->numOfLabel1Samples = numOfLabel1Samples;
		calculateArgs->featureMasksArray_stream_feature = featureMasksArray_device_stream_feature[dev];
		calculateArgs->score = score[dev];
		calculateArgs->device = dev;
		calculateArgs->stream = stream[dev];
		calculateArgs->processor = this;
		
		//cout << streamResult[dev] <<endl;
				
		Task* t = new Task(&asynCalculate, (void*) calculateArgs);
		tp.add_task(t);
	}		
	tp.waitAll();
	tp.destroy_threadpool();
	
	Result* calResult = new Result;		
	calResult->scores = new double[numOfFeatures];
	
	for(int i=0;i<numOfFeatures;i++){
		int dev = i / featuresPerDevice;
		int devRemainder = i % featuresPerDevice;
		int streamId = devRemainder / featuresPerStream;
		int featureId = devRemainder % featuresPerStream;
		if(featureMasksArray_device_stream_feature[dev][streamId][featureId] != true){
			calResult->scores[i] = FEATURE_MASKED;
		}else{
			calResult->scores[i] = score[dev][streamId][featureId];					
		}
		//cout<<dev<<","<<featureIdx<<","<<i<<":"<<score[dev][featureIdx]<<endl;
	}	
		
	for(int i=0; i<numberOfDevices; i++) {
		for(int j=0; j<numberOfStreams; j++){
			free(score[i][j]);
		}
		free(score[i]);
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

Result* GpuAcceleratedProcessor::calculate(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels){
		
	Timer pre("Pre-processing");
	pre.start();
	
	//group samples by label
	int numOfLabel0Samples = 0;
	int numOfLabel1Samples = 0;
	
	for(int j=0; j<numOfSamples; j++)
	{			
		if((int)labels[j]==0){
			numOfLabel0Samples+=1;		
		}else if((int)labels[j]==1){
			numOfLabel1Samples+=1;
		}
	}
		
	//number of array
	//device for GPU
	int numberOfDevices = getNumberOfDevice();
	int featuresPerDevice = getFeaturesPerArray(numOfFeatures, numberOfDevices);
	int numberOfStreams = getNumberOfStreamsPerDevice();
	int featuresPerStream = getFeaturesPerArray(featuresPerDevice, numberOfStreams);
			
	char ***label0SamplesArray_device_stream_feature = (char***)malloc(numberOfDevices * sizeof(char**));
	char ***label1SamplesArray_device_stream_feature = (char***)malloc(numberOfDevices * sizeof(char**));
	bool ***featureMasksArray_device_stream_feature = (bool***)malloc(numberOfDevices * sizeof(bool**));
	int **numberOfFeaturesPerStream = (int**)malloc(numberOfDevices * sizeof(int*));	
	for(int i=0; i<numberOfDevices; i++){
		label0SamplesArray_device_stream_feature[i] = (char**)malloc(featuresPerDevice * sizeof(char*));
		label1SamplesArray_device_stream_feature[i] = (char**)malloc(featuresPerDevice * sizeof(char*));		
		featureMasksArray_device_stream_feature[i] = (bool**)malloc(featuresPerDevice * sizeof(bool*));		
		for(int j=0; j<numberOfStreams;j++){
			label0SamplesArray_device_stream_feature[i][j] = (char*)malloc(featuresPerStream * numOfLabel0Samples * sizeof(char));
			label1SamplesArray_device_stream_feature[i][j] = (char*)malloc(featuresPerStream * numOfLabel1Samples * sizeof(char));
			featureMasksArray_device_stream_feature[i][j] = (bool*)malloc(featuresPerStream * sizeof(bool));
		}
		numberOfFeaturesPerStream[i] = (int*)malloc(featuresPerStream*sizeof(int));
	}

	for(int i=0;i<numberOfDevices;i++){
		for(int j=0;j<numberOfStreams;j++){
			numberOfFeaturesPerStream[i][j] = 0;
		}
	}
	
	if(isDebugEnabled()){
		cout << "featuresPerDevice="<<featuresPerDevice<<", featuresPerStream="<<featuresPerStream<<endl;
	}
	
	for(int i=0;i<numOfFeatures;i++){
		int dev = i / featuresPerDevice;
		int devRemainder = i % featuresPerDevice;
		int streamId = devRemainder / featuresPerStream;
		int featureId = devRemainder % featuresPerStream;
		
		//cout<<"dev="<<dev<<", streamId="<<streamId<<", featureId="<<featureId<<endl;
		
		featureMasksArray_device_stream_feature[dev][streamId][featureId] = featureMask[i];
		
		if(featureMask[i] != true){			
			continue;
		}

		int label0Index=0;
		int label1Index=0;
		
		for(int j=0; j<numOfSamples; j++)
		{
			int index = j*numOfFeatures + i;			
			if(labels[j]==0){
				label0SamplesArray_device_stream_feature[dev][streamId][featureId * numOfLabel0Samples + label0Index]=sampleFeatureMatrix[index];
				label0Index+=1;
			}else if(labels[j]==1){
				label1SamplesArray_device_stream_feature[dev][streamId][featureId * numOfLabel1Samples + label1Index]=sampleFeatureMatrix[index];				
				label1Index+=1;
			}			
		}	
		numberOfFeaturesPerStream[dev][streamId]+=1;
	}
	
	/*
	for(int dev=0; dev<numberOfDevices; dev++) {
		cout << "[GpuAcceleratedProcessor]label0SamplesArray_device_stream_feature["<<dev<<"][0][0]=" << 0+label0SamplesArray_device_stream_feature[dev][0][0] <<", sampleFeatureMatrix[0]="<<0+sampleFeatureMatrix[0]<< endl;
	}
	*/
		
	Result* result = calculateOnDevice(numOfFeatures, 
		label0SamplesArray_device_stream_feature, numOfLabel0Samples,
		label1SamplesArray_device_stream_feature, numOfLabel1Samples, 
		numberOfFeaturesPerStream,
		featureMasksArray_device_stream_feature);

	/*
	for(int i=0; i<numOfFeatures;i++){			
		cout<<"final"<<i<<":"<<result->scores[i]<<endl;
	}	
	*/
		
	//free memory
	for(int i=0; i<numberOfDevices; i++) {
		for(int j=0; j<numberOfStreams; j++){
			free(label0SamplesArray_device_stream_feature[i][j]);
			free(label1SamplesArray_device_stream_feature[i][j]);
			free(featureMasksArray_device_stream_feature[i][j]);	
		}
		free(label0SamplesArray_device_stream_feature[i]);
		free(label1SamplesArray_device_stream_feature[i]);	
		free(featureMasksArray_device_stream_feature[i]);	
		free(numberOfFeaturesPerStream[i]);		
	}
	
	
	free(label0SamplesArray_device_stream_feature);
	free(label1SamplesArray_device_stream_feature);
	free(featureMasksArray_device_stream_feature);	
	free(numberOfFeaturesPerStream);

	return result;
}

