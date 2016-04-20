#include "GpuAcceleratedProcessor.h"
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <string>
#include "threadpool/ThreadPool.h"
#include "utils/Timer.h"

GpuAcceleratedProcessor::GpuAcceleratedProcessor(){
	this->numberOfThreadsPerBlock = 0;
	this->numberOfDevice = 0;
	this->numberOfStreamsPerDevice = 1;
	this->threadPoolEnabled = false;
}

void GpuAcceleratedProcessor::getMemoryInfo(string message){

	if(isDebugEnabled()){		
		size_t free, total;		
		cudaMemGetInfo(&free,&total); 
		cout<<message<<" "<<free/1024<<" KB free of total "<<total/1024<<" KB"<<endl;
	}

}

void GpuAcceleratedProcessor::enableThreadPool(){
	this->threadPoolEnabled = true;
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

Result* GpuAcceleratedProcessor::calculateOnDevice(int numOfFeatures, 
	char*** label0SamplesArray_device_stream_feature, int numOfLabel0Samples,
	char*** label1SamplesArray_device_stream_feature, int numOfLabel1Samples, 
	int** numberOfFeaturesPerStream,
	bool*** featureMasksArray_device_stream_feature,
	bool* success, string* errorMessage){
		
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
	cudaStream_t **stream = (cudaStream_t**)malloc(deviceCount * sizeof(cudaStream_t*));
	for(int i=0; i<deviceCount; i++) {
		stream[i] = (cudaStream_t*)malloc(streamCount * sizeof(cudaStream_t));
	}
				
	for(int dev=0; dev<deviceCount; dev++) {		
		cudaSetDevice(dev);
		for(int i=0; i<streamCount; i++){			
			cudaStreamCreate(&stream[dev][i]);			
		}		
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
	if(isDebugEnabled()){	
		cout<<"wait for streams ready"<<endl;
	}
	if(threadPoolEnabled){
		tp.waitAll();
	}
	
	Timer processing("Processing Time");
	//note that creating stream first can save some time, can further investigate if needed
	processing.start();
	
	calculateOnDeviceWithStream(numberOfFeaturesPerStream,
			label0SamplesArray_device_stream_feature, numOfLabel0Samples,
			label1SamplesArray_device_stream_feature, numOfLabel1Samples,
			featureMasksArray_device_stream_feature,		
			score,			
			stream,
			success, errorMessage);
			
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
		//cout<<dev<<","<<featureId<<","<<i<<":"<<score[dev][streamId][featureId]<<endl;
	}	
	
	for(int d=0; d<numberOfDevices; d++) {			
		cudaSetDevice(d);
		for(int i=0; i<streamCount; i++){			
			cudaError_t streamResult = cudaStreamDestroy(stream[d][i]);	
		}		
		
	}	
	
	for(int i=0; i<numberOfDevices; i++) {		
		for(int j=0; j<numberOfStreams; j++){
			free(score[i][j]);
		}
		free(score[i]);
		free(stream[i]);
	}
	free(score);
	free(stream);
	
	processing.stop();
	totalProcessing.stop();	
	calResult->startTime=processing.getStartTime();
	calResult->endTime=processing.getStopTime();	
	calResult->success=true;
	
	//cout<<"successPerDevice["<<i<<"]"<<successPerDevice[i];
	if (success == false){
		calResult->success = false;		
		if(isDebugEnabled()){
			cout<<errorMessage<<endl;
		}
	}
	
	if(!calResult->success){
		calResult->errorMessage = *errorMessage;
	}
	return calResult;	
			
	/*
		don't forget try stream features in warp size in child classes
	*/
}


Result* GpuAcceleratedProcessor::calculate(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels){
	
	if(getParallelizationType() == PARALLELIZE_ON_FEATURES){
		if(isDebugEnabled()){
			cout<<"parallelize calculation on features"<<endl;
		}	
		return parallelizeCalculationOnFeatures(numOfSamples, numOfFeatures, sampleFeatureMatrix, featureMask, labels);
	}else if(getParallelizationType() == PARALLELIZE_ON_STAGES){
		if(isDebugEnabled()){
			cout<<"parallelize calculation on stages"<<endl;
		}
		
		getMemoryInfo("");
		
		int* packedSampleFeatureMatrix = packSampleFeatureMatrix(numOfSamples, numOfFeatures, sampleFeatureMatrix, featureMask);		
				
		return parallelizeCalculationOnStages(numOfSamples, numOfFeatures, sampleFeatureMatrix, packedSampleFeatureMatrix, featureMask, labels);
	}
	
	return NULL;
}

Result* GpuAcceleratedProcessor::parallelizeCalculationOnFeatures(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels){
		
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
	bool success;
	string errorMessage;
	
	for(int i=0; i<numberOfDevices; i++){
		label0SamplesArray_device_stream_feature[i] = (char**)malloc(featuresPerDevice * sizeof(char*));
		label1SamplesArray_device_stream_feature[i] = (char**)malloc(featuresPerDevice * sizeof(char*));		
		featureMasksArray_device_stream_feature[i] = (bool**)malloc(featuresPerDevice * sizeof(bool*));				
		for(int j=0; j<numberOfStreams;j++){
			label0SamplesArray_device_stream_feature[i][j] = (char*)malloc(featuresPerStream * numOfLabel0Samples * sizeof(char));
			label1SamplesArray_device_stream_feature[i][j] = (char*)malloc(featuresPerStream * numOfLabel1Samples * sizeof(char));
			featureMasksArray_device_stream_feature[i][j] = (bool*)malloc(featuresPerStream * sizeof(bool));
		}
		numberOfFeaturesPerStream[i] = (int*)calloc(featuresPerStream,sizeof(int));
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
		featureMasksArray_device_stream_feature,
		&success, &errorMessage);

	/*
	for(int i=0; i<numOfFeatures;i++){			
		cout<<"[final]"<<i<<":"<<result->scores[i]<<endl;
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

