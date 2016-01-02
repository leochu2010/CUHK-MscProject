#include "GpuProcessor.h"
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

GpuProcessor::GpuProcessor(){
	this->numberOfThreadsPerBlock = 0;
	this->numberOfDevice = 0;
	this->activated = false;	
}

void GpuProcessor::setNumberOfThreadsPerBlock(int numberOfThreadsPerBlock)
{
		this->numberOfThreadsPerBlock = numberOfThreadsPerBlock;
}

int GpuProcessor::getNumberOfThreadsPerBlock()
{
		if (this->numberOfThreadsPerBlock >0){
			return numberOfThreadsPerBlock;
		}else{
			return 1024;
		}
}

void GpuProcessor::setNumberOfDevice(int numberOfDevice){
	this->numberOfDevice = numberOfDevice;
}

int GpuProcessor::getNumberOfDevice(){
	
	if (this->numberOfDevice >0){
			return numberOfDevice;
		}else{
			int deviceCount = 0;
			cudaGetDeviceCount(&deviceCount);
			return deviceCount;
		}
}

int GpuProcessor::getNumberOfFeatureSizeTimesSampleSize2dArrays(int numOfFeatures){
	return this->getNumberOfDevice();
}

Result* GpuProcessor::calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels){
	return new Result;
}

Result* GpuProcessor::calculate(int numOfFeatures, 
		char** label0ProcessingUnitFeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
		char** label1ProcessingUnitFeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
		bool* featureMask){
		
		//can do muti-threading here
			
			
	return new Result;
	
}

