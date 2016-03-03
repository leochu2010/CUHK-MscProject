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

__device__ void pushSampleIdIntoBucket(int sample1Id, int sample2Id, int numOfFeatures, int distance, int kNearest, int* d_distanceBuckets){
		
		int sampleBucketNum = sample1Id * numOfFeatures * (kNearest+1) + (kNearest+1) * distance;
		if(d_distanceBuckets[sampleBucketNum] < kNearest){				
		
			int sampleBucketIdx = atomicAdd(&d_distanceBuckets[sampleBucketNum], 1) + 1;			
			if (sampleBucketIdx > kNearest){
				atomicSub(&d_distanceBuckets[sampleBucketNum], 1);
			}else{
				atomicAdd(&d_distanceBuckets[sampleBucketNum + sampleBucketIdx], sample2Id);
			}
		
		}	
}

__global__ void generateDisatanceBuckets(
		int kNearest,
		int samplePerThread,
		int numOfSamples,			
		int numOfFeatures,		
		int intsPerInstance,
		char* d_labels,		
		int* d_packedSampleFeatureMatrix,		
		int* d_hitDistanceBuckets,
		int* d_missDistanceBuckets		
	){
		
	int sample1Id = gridDim.x * blockIdx.x + blockIdx.y;	

	if(sample1Id >= numOfSamples){
		return;
	}

	/*
	if(threadIdx.x == 0){
		printf("sample1=%d, gridDim.x=%d * blockIdx.x=%d + blockIdx.y=%d\n", sample1Id,gridDim.x,blockIdx.x,blockIdx.y);
	}
	*/
	
	for(int i = 0; i < samplePerThread; i++){
		
		int sample2Id = threadIdx.x * samplePerThread + i;
		
		if(sample2Id <= sample1Id){
			continue;
		}
		
		if(sample2Id >= numOfSamples){
			break;
		}
					
		int distance = 0;
		for(int k = 0; k < intsPerInstance; k++){
			int first = d_packedSampleFeatureMatrix[sample1Id * intsPerInstance + k];
			int second = d_packedSampleFeatureMatrix[sample2Id * intsPerInstance + k];
			int ret = first ^ second;
			for(int l = 0; l < 32; l += 2){
				int diff = (ret >> l) & 3;
				if(diff != 0) distance++; 
			}
		}
		
		if(d_labels[sample1Id] == d_labels[sample2Id]){			
			pushSampleIdIntoBucket(sample1Id, sample2Id, numOfFeatures, distance, kNearest, d_hitDistanceBuckets);
			pushSampleIdIntoBucket(sample2Id, sample1Id, numOfFeatures, distance, kNearest, d_hitDistanceBuckets);
		}else{			
			pushSampleIdIntoBucket(sample1Id, sample2Id, numOfFeatures, distance, kNearest, d_missDistanceBuckets);
			pushSampleIdIntoBucket(sample2Id, sample1Id, numOfFeatures, distance, kNearest, d_missDistanceBuckets);
		}
	}
}

__device__ void findKNearest(int numOfFeatures, int sampleId, int* d_distanceBuckets, int* d_kNearestSampleId, int kNearest){
	int numOfSamples = 0;	
	for(int distance=0; distance<numOfFeatures; distance++){	
			
		int bucket = sampleId * numOfFeatures * (kNearest+1) + (kNearest+1) * distance;
		/*
		if(sampleId == 0){			
			printf("d_hitDistanceBuckets[%d]=%d, %d, %d, %d, %d, %d \n",distance, d_hitDistanceBuckets[bucket], 
			d_hitDistanceBuckets[bucket + 1],
			d_hitDistanceBuckets[bucket + 2],
			d_hitDistanceBuckets[bucket + 3],
			d_hitDistanceBuckets[bucket + 4],
			d_hitDistanceBuckets[bucket + 5]);
		}*/
		
		for(int i=0; i<d_distanceBuckets[bucket];i++){
			int nearSampleId = d_distanceBuckets[bucket + i+1];
			d_kNearestSampleId[sampleId * kNearest + numOfSamples] = nearSampleId;
			numOfSamples += 1;
			/*
			if(sampleId == 0){
				printf("sampleId:%d\n",nearSampleId);
			}*/
			if(numOfSamples == kNearest){
				return;
			}
		}
	}
}

__global__ void weightFeatures(
		int kNearest,
		int numOfFeatures,
		int numOfSamples,
		int* d_kNearestHit,
		int* d_kNearestMiss,
		int* d_hitDistanceBuckets,
		int* d_missDistanceBuckets,
		bool* d_featureMask,
		char* d_sampleFeatureMatrix,
		float* d_weight,
		float* d_finalWeight
	){
		
	int sampleId = gridDim.x * blockIdx.x + blockIdx.y;
	
	//printf("sample=%d, gridDim.x=%d * blockIdx.x=%d + blockIdx.y=%d\n", sampleId,gridDim.x,blockIdx.x,blockIdx.y);	
	
	
	if(sampleId >= numOfSamples){
		return;
	}
	
	findKNearest(numOfFeatures, sampleId, d_hitDistanceBuckets, d_kNearestHit, kNearest);
	findKNearest(numOfFeatures, sampleId, d_missDistanceBuckets, d_kNearestMiss, kNearest);
		
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

Result* GpuAcceleratedReliefFProcessor::parallelizeCalculationOnStages(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, int* packedSampleFeatureMatrix, bool* featureMask, char* labels){
	
	if(isDebugEnabled()){
		cout<<"numOfSamples="<<numOfSamples<<", numOfFeatures="<<numOfFeatures<<endl;
	}
	
	Timer processing("Processing");
	processing.start();
	
	int kNearest = getKNearest();
	
	bool* d_featureMask;
	char* d_sampleFeatureMatrix; 
	int* d_packedSampleFeatureMatrix;	
	int* d_hitDistanceBuckets;
	int* d_missDistanceBuckets;
	char* d_labels;
		
	int intsPerInstance = numOfFeatures / 16 + (numOfFeatures % 16 == 0? 0 : 1);
	
	cudaMalloc(&d_featureMask, numOfFeatures*sizeof(bool));
	cudaMalloc(&d_sampleFeatureMatrix, numOfFeatures * numOfSamples*sizeof(char));
	cudaMalloc(&d_packedSampleFeatureMatrix, intsPerInstance * numOfSamples*sizeof(int));	
	cudaMalloc(&d_labels, numOfSamples*sizeof(char));		
		
	cudaMemcpy(d_featureMask, featureMask, numOfFeatures*sizeof(bool),cudaMemcpyHostToDevice);
	cudaMemcpy(d_sampleFeatureMatrix, sampleFeatureMatrix, numOfFeatures*numOfSamples*sizeof(char),cudaMemcpyHostToDevice);
	cudaMemcpy(d_packedSampleFeatureMatrix, packedSampleFeatureMatrix, intsPerInstance * numOfSamples*sizeof(int),cudaMemcpyHostToDevice);	
	cudaMemcpy(d_labels, labels, numOfSamples*sizeof(char),cudaMemcpyHostToDevice);
	
	cudaMalloc(&d_hitDistanceBuckets, numOfSamples * numOfFeatures * (kNearest+1) * sizeof(int));
	cudaMalloc(&d_missDistanceBuckets, numOfSamples * numOfFeatures * (kNearest+1) * sizeof(int));
	cudaMemset(d_hitDistanceBuckets, 0, numOfSamples * numOfFeatures * (kNearest+1) * sizeof(int));
	cudaMemset(d_missDistanceBuckets, 0, numOfSamples * numOfFeatures * (kNearest+1) * sizeof(int));

		
	int grid2d = (int)ceil(pow(numOfSamples,1/2.));
	int threadSize = getNumberOfThreadsPerBlock();
	
	int samplePerThread = (int)ceil(((float)numOfSamples)/threadSize);
	int maxSampleId = (int)ceil(((float)numOfSamples)/2);
	
	dim3 gridSize(grid2d,grid2d);
	
	if(isDebugEnabled()){
		cout<<"generate distance buckets"<<endl;
	}	
	generateDisatanceBuckets<<<gridSize, threadSize>>>(	
		kNearest,
		samplePerThread,
		numOfSamples,			
		numOfFeatures,		
		intsPerInstance,
		d_labels,		
		d_packedSampleFeatureMatrix,		
		d_hitDistanceBuckets,
		d_missDistanceBuckets
		);
	cudaDeviceSynchronize();
	
	cudaFree(d_labels);
	
	int* d_kNearestHit;
	int* d_kNearestMiss;
	cudaMalloc(&d_kNearestHit, kNearest * numOfSamples*sizeof(int));
	cudaMalloc(&d_kNearestMiss, kNearest * numOfSamples*sizeof(int));
		
	float* finalWeight = (float*)calloc(numOfFeatures,sizeof(float));
	float* d_weight;
	float* d_finalWeight;
	
	cudaMalloc(&d_weight, numOfSamples*numOfFeatures*sizeof(float));
	cudaMemset(d_weight, 0, numOfSamples*numOfFeatures*sizeof(float));
	
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
		d_hitDistanceBuckets,
		d_missDistanceBuckets,
		d_featureMask,
		d_sampleFeatureMatrix,
		d_weight,
		d_finalWeight
	);
		
	cudaDeviceSynchronize();
	
	cudaMemcpy(finalWeight, d_finalWeight, numOfFeatures*sizeof(float), cudaMemcpyDeviceToHost);
			
	if(isDebugEnabled()){
		cout<<"generate result"<<endl;
	}
	Result* result = new Result;
	result->scores = new double[numOfFeatures];
	int divider = numOfSamples * kNearest;
	for(int i=0;i<numOfFeatures;i++){
		result->scores[i] = finalWeight[i]/divider;
	}
	result->success = true;	
		
	free(finalWeight);
	
	cudaFree(d_hitDistanceBuckets);
	cudaFree(d_missDistanceBuckets);
	cudaFree(d_featureMask);
	cudaFree(d_sampleFeatureMatrix);
	cudaFree(d_packedSampleFeatureMatrix);	
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

