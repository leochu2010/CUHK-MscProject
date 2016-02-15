#include "GpuAcceleratedReliefFProcessor.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "utils/Timer.h"

using namespace std;

GpuAcceleratedReliefFProcessor::GpuAcceleratedReliefFProcessor(int kNearest){
	parallelizationType = PARALLELIZE_ON_STAGES;
	kNearestInstance = kNearest;
}

int GpuAcceleratedReliefFProcessor::getKNearest(){
	return kNearestInstance;
}

__global__ void generateDisatanceMatrix(
		int samplePerThread,
		int numOfSamples,		
		int numOfFeatures,
		bool* d_featureMask,
		char* d_sampleFeatureMatrix,
		int* d_distanceMatrix
	){
		
	int sample1Id = gridDim.x * blockIdx.x + blockIdx.y;	

	if(sample1Id > numOfSamples){
		return;
	}	

	/*
	if(threadIdx.x == 0){
		printf("sample1=%d, gridDim.x=%d * blockIdx.x=%d + blockIdx.y=%d\n", sample1Id,gridDim.x,blockIdx.x,blockIdx.y);
	}
	*/	
	
	for(int i = 0; i < samplePerThread; i++){
		int distance = 0;
		int sample2Id = threadIdx.x * samplePerThread + i;
		if(sample2Id > numOfSamples){
			continue;
		}
		
		for(int featureId = 0; featureId < numOfFeatures; featureId ++ ){
			
			if(d_featureMask[featureId] != true){
				continue;
			}
			
			if(d_sampleFeatureMatrix[sample1Id * numOfFeatures + featureId] != d_sampleFeatureMatrix[sample2Id * numOfFeatures + featureId]){
				distance += 1;
			}
			
		}
		d_distanceMatrix[sample1Id * numOfSamples + sample2Id] = distance;
		//printf("d_distanceMatrix[%d]=%d \n",sample1Id * numOfSamples + sample2Id,d_distanceMatrix[sample1Id * numOfSamples + sample2Id]);
	}
	
}

__device__ int getMinDistanceIndex(int* sampleDistanceMatrix, bool* ignore, int numOfSamples, int numOfFeatures, char *labels, char label, bool sameLabel, int sample1Id){
	
	int minDistance = numOfFeatures;
	int minDistanceIndex = -1;
		
	for(int i=0; i<numOfSamples; i++){
		
		if(sameLabel){
			if(labels[i] != label){
				continue;			
			}
		}else{
			if(labels[i] == label){			
				continue;			
			}
		}
		
		if(ignore[sample1Id * numOfSamples + i]){
			continue;
		}
		
		if(sampleDistanceMatrix[sample1Id * numOfSamples + i] < minDistance){
			//printf("sampleDistanceMatrix[%d]=%d \n",sample1Id * numOfSamples + i,sampleDistanceMatrix[sample1Id * numOfSamples + i]);
			minDistanceIndex = i;
			minDistance = sampleDistanceMatrix[sample1Id * numOfSamples + i];
		}
	}
	
	//printf("getMinDistanceIndex = %d, label=%d\n",minDistanceIndex, label);
	return minDistanceIndex;	
}

__global__ void findKNearestSamples(
		int kNearest,
		int samplePerThread,
		int numOfSamples,
		int numOfFeatures,
		bool* d_ignoreHit,
		bool* d_ignoreMiss,
		int* d_sampleDistanceMatrix,
		char* d_labels,
		int* d_kNearestHit,
		int* d_kNearestMiss		
	){
		
	int sample1Id = gridDim.x * blockIdx.x + blockIdx.y;	
	
	//printf("sample1=%d, gridDim.x=%d * blockIdx.x=%d + blockIdx.y=%d\n", sample1Id,gridDim.x,blockIdx.x,blockIdx.y);	
	
	if(sample1Id > numOfSamples){
		return;
	}
	
	d_ignoreMiss[sample1Id * numOfSamples + sample1Id] = true;
	d_ignoreHit[sample1Id * numOfSamples + sample1Id] = true;
	
	bool sameLabel = true;
	for(int i=0; i<kNearest; i++){
		int minDistanceId = getMinDistanceIndex(d_sampleDistanceMatrix, d_ignoreHit, numOfSamples, numOfFeatures, d_labels, d_labels[sample1Id], sameLabel, sample1Id);
		d_ignoreHit[sample1Id * numOfSamples + minDistanceId] = true;
		d_kNearestHit[sample1Id * kNearest + i] = minDistanceId;
	}
	
	sameLabel = false;
	for(int i=0; i<kNearest; i++){
		int minDistanceId = getMinDistanceIndex(d_sampleDistanceMatrix, d_ignoreMiss, numOfSamples, numOfFeatures, d_labels, d_labels[sample1Id], sameLabel, sample1Id);
		d_ignoreMiss[sample1Id * numOfSamples + minDistanceId] = true;
		d_kNearestMiss[sample1Id * kNearest + i] = minDistanceId;
	}	
	
}

__global__ void weightFeatures(
		int kNearest,
		int numOfFeatures,
		int numOfSamples,
		int* d_kNearestHit,
		int* d_kNearestMiss,
		bool* d_featureMask,
		char* d_sampleFeatureMatrix,
		float* d_weight,
		float* d_finalWeight
	){
		
	int sampleId = gridDim.x * blockIdx.x + blockIdx.y;
	
	//printf("sample=%d, gridDim.x=%d * blockIdx.x=%d + blockIdx.y=%d\n", sampleId,gridDim.x,blockIdx.x,blockIdx.y);	
	
	
	if(sampleId > numOfSamples){
		return;
	}
	
	for(int j=0; j<numOfFeatures; j++){
		d_weight[sampleId * numOfFeatures + j] = 0;
	}
	
	for(int i=0; i<kNearest; i++){
		int hitSampleId = d_kNearestHit[sampleId * kNearest + i];
		int missSampleId = d_kNearestMiss[sampleId * kNearest + i];
		
		for(int j=0; j<numOfFeatures; j++){
			
			if(d_featureMask[j] != true){
				continue;
			}						
			
			char feature = d_sampleFeatureMatrix[sampleId * numOfFeatures + j];
			char hitFeature = d_sampleFeatureMatrix[hitSampleId * numOfFeatures + j];
			char missFeature = d_sampleFeatureMatrix[missSampleId * numOfFeatures + j];
			
			if (feature != hitFeature){
				d_weight[sampleId * numOfFeatures + j] -= 1;
			}
			
			if (feature != missFeature){
				d_weight[sampleId * numOfFeatures + j] += 1;
			}
		}
	}
	
	for(int i=0; i<numOfFeatures; i++){						
		atomicAdd(&d_finalWeight[i],d_weight[sampleId * numOfFeatures + i]);
	}
	
}

Result* GpuAcceleratedReliefFProcessor::parallelizeCalculationOnStages(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels){
	
	if(isDebugEnabled()){
		cout<<"numOfSamples="<<numOfSamples<<", numOfFeatures="<<numOfFeatures<<endl;
	}
	
	Timer processing("Processing");
	processing.start();
	
	bool* d_featureMask;
	char* d_sampleFeatureMatrix; 
	int* d_distanceMatrix;	
	
	cudaMalloc(&d_featureMask, numOfFeatures*sizeof(bool));
	cudaMalloc(&d_sampleFeatureMatrix, numOfFeatures * numOfSamples*sizeof(char));
	cudaMalloc(&d_distanceMatrix, numOfSamples * numOfSamples*sizeof(int));
	
	cudaMemcpy(d_featureMask, featureMask, numOfFeatures*sizeof(bool),cudaMemcpyHostToDevice);
	cudaMemcpy(d_sampleFeatureMatrix, sampleFeatureMatrix, numOfFeatures*numOfSamples*sizeof(char),cudaMemcpyHostToDevice);	
		
	int grid2d = (int)ceil(pow(numOfSamples,1/2.));
	int threadSize = getNumberOfThreadsPerBlock();
	
	int samplePerThread = (int)ceil(((float)numOfSamples)/threadSize);
	
	dim3 gridSize(grid2d,grid2d);
	
	if(isDebugEnabled()){
		cout<<"calculate distance matrix"<<endl;
	}	
	generateDisatanceMatrix<<<gridSize, threadSize>>>(
		samplePerThread,
		numOfSamples,
		numOfFeatures,
		d_featureMask,
		d_sampleFeatureMatrix,
		d_distanceMatrix
		);
	cudaDeviceSynchronize();
			
	int kNearest = getKNearest();
			
	bool* d_ignoreHit;
	bool* d_ignoreMiss;
	char* d_labels;
	int* d_kNearestHit;
	int* d_kNearestMiss;	
	cudaMalloc(&d_ignoreHit, numOfSamples*numOfSamples*sizeof(bool));
	cudaMalloc(&d_ignoreMiss, numOfSamples*numOfSamples*sizeof(bool));	
	cudaMalloc(&d_labels, numOfSamples*sizeof(char));		
	cudaMalloc(&d_kNearestHit, kNearest * numOfSamples*sizeof(int));
	cudaMalloc(&d_kNearestMiss, kNearest * numOfSamples*sizeof(int));	
	
	if(isDebugEnabled()){
		cout<<"find the "<<kNearest<<" nearest samples"<<endl;
	}	
	
	bool* ignoreHit = (bool*)calloc(numOfSamples*numOfSamples,sizeof(bool));
	bool* ignoreMiss = (bool*)calloc(numOfSamples*numOfSamples,sizeof(bool));
	
	cudaMemcpy(d_labels, labels, numOfSamples*sizeof(char),cudaMemcpyHostToDevice);
	cudaMemcpy(d_ignoreHit, ignoreHit, numOfSamples*numOfSamples*sizeof(bool),cudaMemcpyHostToDevice);
	cudaMemcpy(d_ignoreMiss, ignoreMiss, numOfSamples*numOfSamples*sizeof(bool),cudaMemcpyHostToDevice);
	
	findKNearestSamples<<<gridSize, 1>>>(
		kNearest,
		samplePerThread,
		numOfSamples,
		numOfFeatures,
		d_ignoreHit,
		d_ignoreMiss,
		d_distanceMatrix,
		d_labels,
		d_kNearestHit,
		d_kNearestMiss		
	);
	cudaDeviceSynchronize();
	cudaFree(d_distanceMatrix);
	cudaFree(d_labels);
		
	float* finalWeight = (float*)calloc(numOfFeatures,sizeof(float));	
	
	float* d_weight;
	float* d_finalWeight;	
	cudaMalloc(&d_weight, numOfSamples*numOfFeatures*sizeof(float));
	cudaMalloc(&d_finalWeight, numOfFeatures*sizeof(float));	
	cudaMemcpy(d_finalWeight, finalWeight, numOfFeatures*sizeof(float),cudaMemcpyHostToDevice);
	
	if(isDebugEnabled()){
		cout<<"weight features"<<endl;
	}	
		
	weightFeatures<<<gridSize,1>>>(
		kNearest,
		numOfFeatures,
		numOfSamples,
		d_kNearestHit,
		d_kNearestMiss,
		d_featureMask,
		d_sampleFeatureMatrix,
		d_weight,
		d_finalWeight
	);
		
	cudaDeviceSynchronize();
	
	free(ignoreHit);
	free(ignoreMiss);
	cudaFree(d_ignoreHit);
	cudaFree(d_ignoreMiss);
		
	cudaMemcpy(finalWeight, d_finalWeight, numOfFeatures*sizeof(float), cudaMemcpyDeviceToHost);
			
	if(isDebugEnabled()){
		cout<<"generate result"<<endl;
	}
	Result* result = new Result;
	result->scores = new double[numOfFeatures];
	for(int i=0;i<numOfFeatures;i++){
		result->scores[i] = finalWeight[i]/(numOfSamples * kNearest);
	}
	result->success = true;	
		
	free(finalWeight);
	
	cudaFree(d_featureMask);
	cudaFree(d_sampleFeatureMatrix);
	cudaFree(d_distanceMatrix);	
	cudaFree(d_labels);
	cudaFree(d_kNearestHit);
	cudaFree(d_kNearestMiss);
	cudaFree(d_weight);
	cudaFree(d_finalWeight);
	
	processing.stop();
	result->startTime=processing.getStartTime();
	result->endTime=processing.getStopTime();	
	
	return result;
}

