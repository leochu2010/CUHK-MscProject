#include "GpuAcceleratedMutualInformationProcessor.h"

#include <iostream>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

using namespace std;

GpuAcceleratedMutualInformationProcessor::GpuAcceleratedMutualInformationProcessor(){
	parallelizationType = PARALLELIZE_ON_FEATURES;
}

__global__ void calculateMutualInformation(
	char *d_firstVector, char *d_secondVector, 
	int vectorLength, int vectorLengthPerThread,
	bool *d_featureMask,
	size_t numOfFeatures,	
	double *d_score, int device,
	int* d_excpetion)
{
	
	//gridDim.x = gridDim.y as it is 2d 
	//idx is feature index of a block on this device
	int idx = gridDim.x * blockIdx.x + blockIdx.y;
	
	/*
	if(threadIdx.x == 0){
		printf("idx=%d, blockDim.x=%d, threadIdx.x=%d",idx, blockDim.x, threadIdx.x);
	}
	*/
	
	if (idx >= numOfFeatures){		
		/*
		if(threadIdx.x == 0){
			printf("device[%d]:idx=%d >= %d \n",device,idx,numOfFeatures);
		}		
		*/
		return;
	}
	
	if (d_featureMask[idx] != true){			
		/*
		if(threadIdx.x == 0){
			printf("device[%d]:idx=%d, d_featureMask=false\n",device,idx);
		}
		*/
		return;
	}
	
	__shared__ float mutualInformation;	
	
	__shared__ int firstNumStates;
	__shared__ int secondNumStates;
	__shared__ int jointNumStates;	
	
	__shared__ int firstVectorMinVal, firstVectorMaxVal;
	__shared__ int secondVectorMinVal, secondVectorMaxVal;
	
	if(threadIdx.x == 0){
		mutualInformation= 0.0;
		firstVectorMinVal = d_firstVector[vectorLength * idx];
		firstVectorMaxVal = d_firstVector[vectorLength * idx];
		secondVectorMinVal = d_secondVector[vectorLength * idx];
		secondVectorMaxVal = d_secondVector[vectorLength * idx];
	}
	
	__syncthreads();
	
	if(vectorLengthPerThread*(threadIdx.x) < vectorLength){
		for(int i=vectorLengthPerThread*(threadIdx.x); i<vectorLengthPerThread*(threadIdx.x+1) && i<vectorLength; i++){
			int firstVectorCurrentValue = (int)(d_firstVector[vectorLength * idx + i]);
			atomicMin(&firstVectorMinVal, firstVectorCurrentValue);
			atomicMax(&firstVectorMaxVal, firstVectorCurrentValue);
			d_firstVector[vectorLength * idx + i] = firstVectorCurrentValue;
			
			int secondVectorCurrentValue = (int)(d_secondVector[vectorLength * idx + i]);
			atomicMin(&secondVectorMinVal, secondVectorCurrentValue);
			atomicMax(&secondVectorMaxVal, secondVectorCurrentValue);
			d_secondVector[vectorLength * idx + i] = secondVectorCurrentValue;
		}		
	}
	
	__syncthreads();
	if(threadIdx.x == 0){
		firstNumStates = (firstVectorMaxVal - firstVectorMinVal) + 1;
		secondNumStates = (secondVectorMaxVal - secondVectorMinVal) + 1;
		jointNumStates = firstNumStates * secondNumStates;
		if(firstNumStates > 5){
			d_excpetion[0] = -1;
		}
		if(secondNumStates > 5){
			d_excpetion[0] = -2;
		}
		if(jointNumStates > 25){
			d_excpetion[0] = -3;
		}
				
		/*
		if(idx==0){
			printf("device[%d]:idx=%d, d_excpetion = %d < 0, firstNumStates=%d, secondNumStates=%d, jointNumStates=%d \n",device,idx,d_excpetion[0],firstNumStates,secondNumStates,jointNumStates);
		}
		*/
	}
	
	if(vectorLengthPerThread*(threadIdx.x) < vectorLength){
		for(int i=vectorLengthPerThread*(threadIdx.x); i<vectorLengthPerThread*(threadIdx.x+1) && i<vectorLength; i++){    
			d_firstVector[vectorLength * idx + i] = d_firstVector[vectorLength * idx + i] - firstVectorMinVal;
			d_secondVector[vectorLength * idx + i] = d_secondVector[vectorLength * idx + i] - secondVectorMinVal;
		}
	}
	
	__syncthreads();
	
	if(d_excpetion[0] < 0){
		/*
		if(threadIdx.x == 0){
			printf("device[%d]:idx=%d, d_excpetion = %d < 0, firstNumStates=%d, secondNumStates=%d, jointNumStates=%d \n",device,idx,d_excpetion[0],firstNumStates,secondNumStates,jointNumStates);
		}
		*/
		return;
	}	
		
	__shared__ int firstStateCounts[5];
	__shared__ int secondStateCounts[5];
	__shared__ int jointStateCounts[25];
	
	__shared__ double firstStateProbs[5];
	__shared__ double secondStateProbs[5];
	__shared__ double jointStateProbs[25];
	
	if(threadIdx.x < 5){
		firstStateCounts[threadIdx.x] = 0;
		secondStateCounts[threadIdx.x] = 0;		
	
		firstStateProbs[threadIdx.x] = 0.0;
		secondStateProbs[threadIdx.x] = 0.0;
	}
	
	if(threadIdx.x < 25){
		jointStateCounts[threadIdx.x] = 0;
		jointStateProbs[threadIdx.x] = 0.0;
	}
			
	__syncthreads();
	
	/* Optimised for number of FP operations now O(states) instead of O(vectorLength) */
	if(vectorLengthPerThread*(threadIdx.x) < vectorLength){		
		for(int i=vectorLengthPerThread*(threadIdx.x); i<vectorLengthPerThread*(threadIdx.x+1) && i<vectorLength; i++){
			atomicAdd(&firstStateCounts[d_firstVector[vectorLength * idx + i]], 1);
			atomicAdd(&secondStateCounts[d_secondVector[vectorLength * idx + i]], 1);
			atomicAdd(&jointStateCounts[d_secondVector[vectorLength * idx + i] * firstNumStates + d_firstVector[vectorLength * idx + i]], 1);
		}		
	}	
	
	__shared__ int firstNumStatesPerThread;
	__shared__ int secondNumStatesPerThread;
	__shared__ int jointNumStatesPerThread;
	
	//blockDim: number of threads in a block
	if (threadIdx.x == 0){
		firstNumStatesPerThread = ceil((float)firstNumStates / blockDim.x);
		secondNumStatesPerThread = ceil((float)secondNumStates / blockDim.x);
		jointNumStatesPerThread = ceil((float)jointNumStates / blockDim.x);
	}
	
	__syncthreads();
	
	if(firstNumStatesPerThread*(threadIdx.x) < firstNumStates){
		double length = vectorLength;
		for(int i=firstNumStatesPerThread*(threadIdx.x); i<firstNumStatesPerThread*(threadIdx.x+1) && i<firstNumStates; i++){
			firstStateProbs[i] = firstStateCounts[i] / length;
		}
	}
	
	if(secondNumStatesPerThread*(threadIdx.x) < secondNumStates){
		double length = vectorLength;
		for(int i=secondNumStatesPerThread*(threadIdx.x); i<secondNumStatesPerThread*(threadIdx.x+1) && i<secondNumStates; i++){
			secondStateProbs[i] = secondStateCounts[i] / length;
		}
	}
	
	if(jointNumStatesPerThread*(threadIdx.x) < jointNumStates){
		double length = vectorLength;
		for(int i=jointNumStatesPerThread*(threadIdx.x); i<jointNumStatesPerThread*(threadIdx.x+1) && i<jointNumStates; i++){
			jointStateProbs[i] = jointStateCounts[i] / length;
		}
	}
	
	/*
	** I(X;Y) = sum sum p(xy) * log (p(xy)/p(x)p(y))
	*/	
	if(jointNumStatesPerThread*(threadIdx.x) < jointNumStates){
		for(int i=jointNumStatesPerThread*(threadIdx.x); i<jointNumStatesPerThread*(threadIdx.x+1) && i<jointNumStates; i++){
			int firstIndex = i % firstNumStates;
			int secondIndex = i / firstNumStates;
			if ((jointStateProbs[i] > 0) && (firstStateProbs[firstIndex] > 0) && (secondStateProbs[secondIndex] > 0))
			{
			  /*double division is probably more stable than multiplying two small numbers together
			  ** mutualInformation += jointStateProbs[i] * log(jointStateProbs[i] / (firstStateProbs[firstIndex] * secondStateProbs[secondIndex]));
			  */
			  double addMutualInformation = jointStateProbs[i] * log(jointStateProbs[i] / firstStateProbs[firstIndex] / secondStateProbs[secondIndex]);
			  atomicAdd(&mutualInformation, addMutualInformation);
			}
		}
	}	
	//mutualInformation /= log(2.0);	
	
	__syncthreads();
	
	if (threadIdx.x == 0){
		d_score[idx] = mutualInformation;		
		/*
		printf("device[%d]:d_score[%d]=%f\n",device,idx,mutualInformation);
		*/
	}
}

void GpuAcceleratedMutualInformationProcessor::calculateOnDeviceWithStream(int** numberOfFeaturesPerStream,
			char*** label0SamplesArray_stream_feature, int numOfLabel0Samples,
			char*** label1SamplesArray_stream_feature, int numOfLabel1Samples,
			bool*** featureMasksArray_stream_feature,		
			double*** score,			
			cudaStream_t** streams,
			bool* success, string* errorMessage){
			
	*success = true;
			
	int streamCount = getNumberOfStreamsPerDevice();
	int deviceCount = getNumberOfDevice();
			
	if (numOfLabel0Samples != numOfLabel1Samples){
		for(int d=0; d<deviceCount; d++){
			for(int i=0; i<streamCount; i++){
				for(int j=0; j<numberOfFeaturesPerStream[d][i];j++){
					score[d][i][j]=INVALID_FEATURE;
				}
			}
		}
		*success = false;
		*errorMessage = "numbers of label 0 and 1 samples are not the same";
		if(this->isDebugEnabled()){
			cout << "return: numbers of label 0 and 1 samples are not the same"<<endl;
		}
		return;
	}
			
	/*
	if(device == 1){
		cout<<"[GpuAcceleratedPValueProcessor]"<<"numberOfFeaturesPerStream[0]="<<numberOfFeaturesPerStream[0]<<", label0SamplesArray_stream_feature[0][0]="<<0+label0SamplesArray_stream_feature[0][0]<<endl;	
	}
	*/
	
			
	int threadSize = getNumberOfThreadsPerBlock();
	size_t samplesPerThread = ceil((numOfLabel0Samples/(float)(threadSize)));	
	
	//copy data from main memory to GPU		
	
	char *d_label0Array[deviceCount][streamCount];
	char *d_label1Array[deviceCount][streamCount];
	double *d_score[deviceCount][streamCount];
	bool *d_featureMask[deviceCount][streamCount];
	int *d_exception[deviceCount][streamCount];
		
	if(this->isDebugEnabled()){
		cout << "copy to GPU"<<endl;
	}
	
	int ***exception = (int***)malloc(deviceCount * sizeof(int**));
	for(int d=0; d<deviceCount; d++){
		exception[d] = (int**)malloc(streamCount * sizeof(int*));	
	}
	
	getMemoryInfo("before cudaMalloc");

	
	for(int device=0; device<deviceCount; device++){
		cudaSetDevice(device);	
		int maxFeaturesPerStream = numberOfFeaturesPerStream[device][0];
		
		for(int i=0; i<streamCount; i++){		
			cudaMalloc(&d_label0Array[device][i],maxFeaturesPerStream*numOfLabel0Samples*sizeof(char));
			cudaMalloc(&d_label1Array[device][i],maxFeaturesPerStream*numOfLabel1Samples*sizeof(char));
			cudaMalloc(&d_score[device][i],maxFeaturesPerStream*sizeof(double));
			cudaMalloc(&d_featureMask[device][i],maxFeaturesPerStream*sizeof(bool));
			cudaMalloc(&d_exception[device][i],sizeof(int));		
			
			if(isDebugEnabled()){
				cout<<"device:"<<device<<", steam "<<i<<" cuda malloc"<<endl;
			}
			
			getMemoryInfo("after cudaMalloc");	
			
			exception[device][i] = (int*)malloc(sizeof(int));		
			exception[device][i][0] = 0;
			
			if(isDebugEnabled()){
				cout <<"steam "<<i<<" cuda malloc"<<endl;
			}		
		}	
	}
	
	for(int device=0; device<deviceCount; device++){
		cudaSetDevice(device);	
		
		for(int i=0; i<streamCount; i++){
			int features = numberOfFeaturesPerStream[device][i];
			cudaMemcpyAsync(d_label0Array[device][i],label0SamplesArray_stream_feature[device][i],features*numOfLabel0Samples*sizeof(char),cudaMemcpyHostToDevice,streams[device][i]);
			cudaMemcpyAsync(d_label1Array[device][i],label1SamplesArray_stream_feature[device][i],features*numOfLabel1Samples*sizeof(char),cudaMemcpyHostToDevice,streams[device][i]);
			cudaMemcpyAsync(d_score[device][i],score[device][i],features*sizeof(double),cudaMemcpyHostToDevice,streams[device][i]);
			cudaMemcpyAsync(d_featureMask[device][i],featureMasksArray_stream_feature[device][i],features*sizeof(bool),cudaMemcpyHostToDevice,streams[device][i]);		
			cudaMemcpyAsync(d_exception[device][i],exception[device][i],sizeof(int),cudaMemcpyHostToDevice,streams[device][i]);
		}
	}
	
	for(int dev=0; dev<deviceCount; dev++){
		cudaSetDevice(dev);
		cudaDeviceSynchronize();
 	}
	
	for(int device=0; device<deviceCount; device++){
		
		cudaSetDevice(device);	
		int maxFeaturesPerStream = numberOfFeaturesPerStream[device][0];
		
		//calculate		
		int grid2d = (int)ceil(pow(maxFeaturesPerStream,1/2.));	
		if(this->isDebugEnabled()){
			cout<<"maxFeaturesPerStream="<<maxFeaturesPerStream<<",grid2d="<<grid2d<<",numOfLabel0Samples="<<numOfLabel0Samples<<",numOfLabel1Samples="<<numOfLabel1Samples<<",threadSize="<<threadSize<<endl;
		}
		
		dim3 gridSize(grid2d,grid2d);
		
		
		for(int i=0; i<streamCount; i++){
			calculateMutualInformation<<<gridSize, threadSize, 0, streams[device][i]>>>(
					d_label0Array[device][i], d_label1Array[device][i], 
					numOfLabel0Samples, samplesPerThread,
					d_featureMask[device][i],
					numberOfFeaturesPerStream[device][i],
					d_score[device][i], device, d_exception[device][i]);
					
			if(this->isDebugEnabled()){		
				cout<<"device:"<<device<<", stream:"<<i<<" cudaPeekAtLastError:"<<cudaGetErrorString(cudaPeekAtLastError())<<endl;
			}
		}
	}
	
	for(int dev=0; dev<deviceCount; dev++){
		cudaSetDevice(dev);
		cudaDeviceSynchronize();
 	}

	for(int device=0; device<deviceCount; device++){
		
		cudaSetDevice(device);	
		//copy result from GPU to main memory
		for(int i=0; i<streamCount; i++){
			int features = numberOfFeaturesPerStream[device][i];
			cudaMemcpyAsync(score[device][i], d_score[device][i], features*sizeof(double), cudaMemcpyDeviceToHost,streams[device][i]);
			cudaMemcpyAsync(exception[device][i], d_exception[device][i], features*sizeof(int), cudaMemcpyDeviceToHost,streams[device][i]);		
		}
	}
	if(this->isDebugEnabled()){		
		cout<<"copied data from device to host"<<endl;
	}

	for(int dev=0; dev<deviceCount; dev++){
		cudaSetDevice(dev);
		cudaDeviceSynchronize();
 	}

	for(int device=0; device<deviceCount; device++){
		for(int i=0; i<streamCount; i++){
			
			if(exception[device][i][0] < 0){
				*success = false;
				*errorMessage = "firstNumStates/secondNumStates/jointNumStates is too big";
				if(this->isDebugEnabled()){
					cout<<"exception[i]"<<exception[device][i][0]<<endl;
					cout<<"error found"<<endl;
				}
			}				
		}
	}
		
	//free cuda memory	
	for(int device=0; device<deviceCount; device++){
		for(int i=0; i<streamCount; i++){
			if(isDebugEnabled()){
				cout<<"cudafree resources"<<endl;
			}		
			cudaFree(d_label1Array[device][i]);
			cudaFree(d_label0Array[device][i]);
			cudaFree(d_score[device][i]);
			cudaFree(d_featureMask[device][i]);		
			cudaFree(d_exception[device][i]);		
			free(exception[device][i]);
		}		
	}
	
	for(int device=0; device<deviceCount; device++){
		free(exception[device]);
	}			
	
	free(exception);
	
}	