#include "GpuAcceleratedReliefFBucketSortProcessor.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "utils/Timer.h"

using namespace std;

GpuAcceleratedReliefFBucketSortProcessor::GpuAcceleratedReliefFBucketSortProcessor(int kNearest){
	parallelizationType = PARALLELIZE_ON_STAGES;
	kNearestInstance = kNearest;
}

int GpuAcceleratedReliefFBucketSortProcessor::getKNearest(){
	return kNearestInstance;
}

__device__ void gpu_pushSampleIdIntoBucket(int sample1Id, int sample2Id, int numOfFeatures, int distance, int kNearest, int* d_distanceBuckets){
		
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

__global__ void gpu_generateDisatanceBuckets(
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
			/*
			if(threadIdx.x == 0 && k==0){
				printf("first=%d, second=%d\n", first, second);
			}*/
			int ret = first ^ second;
			for(int l = 0; l < 32; l += 2){
				int diff = (ret >> l) & 3;
				if(diff != 0) distance++; 
			}
		}
		
		if(d_labels[sample1Id] == d_labels[sample2Id]){			
			gpu_pushSampleIdIntoBucket(sample1Id, sample2Id, numOfFeatures, distance, kNearest, d_hitDistanceBuckets);
			gpu_pushSampleIdIntoBucket(sample2Id, sample1Id, numOfFeatures, distance, kNearest, d_hitDistanceBuckets);
		}else{			
			gpu_pushSampleIdIntoBucket(sample1Id, sample2Id, numOfFeatures, distance, kNearest, d_missDistanceBuckets);
			gpu_pushSampleIdIntoBucket(sample2Id, sample1Id, numOfFeatures, distance, kNearest, d_missDistanceBuckets);
		}
	}
}

__device__ void gpu_findKNearestFromBuckets(int numOfFeatures, int sampleId, int* d_distanceBuckets, int* d_kNearestSampleId, int kNearest){
	int numOfSamples = 0;	
	for(int distance=0; distance<numOfFeatures; distance++){	
			
		int bucket = sampleId * numOfFeatures * (kNearest+1) + (kNearest+1) * distance;
		
		/*
		if(sampleId == 0){			
			printf("d_distanceBuckets[%d]=%d, %d, %d, %d, %d, %d \n",distance, d_distanceBuckets[bucket], 
			d_distanceBuckets[bucket + 1],
			d_distanceBuckets[bucket + 2],
			d_distanceBuckets[bucket + 3],
			d_distanceBuckets[bucket + 4],
			d_distanceBuckets[bucket + 5]);
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

__global__ void gpu_weightFeaturesFromBuckets(
		int kNearest,
		int numOfFeatures,
		int numOfSamples,
		int intsPerInstance,
		int* d_kNearestHit,
		int* d_kNearestMiss,
		int* d_hitDistanceBuckets,
		int* d_missDistanceBuckets,
		bool* d_featureMask,
		int* d_packedSampleFeatureMatrix,
		float* d_weight,
		float* d_finalWeight
	){
		
	int sampleId = gridDim.x * blockIdx.x + blockIdx.y;
	
	//printf("sample=%d, gridDim.x=%d * blockIdx.x=%d + blockIdx.y=%d\n", sampleId,gridDim.x,blockIdx.x,blockIdx.y);	
	
	
	if(sampleId >= numOfSamples){
		return;
	}
	
	gpu_findKNearestFromBuckets(numOfFeatures, sampleId, d_hitDistanceBuckets, d_kNearestHit, kNearest);
	gpu_findKNearestFromBuckets(numOfFeatures, sampleId, d_missDistanceBuckets, d_kNearestMiss, kNearest);
	
	/*
	if(sampleId == 999){
		for(int i=0;i<kNearest;i++){
			printf("k=%d, hitSampleId=%d, missSampleId=%d\n", i, d_kNearestHit[sampleId * kNearest+i], d_kNearestMiss[sampleId * kNearest+i]);
		}
	}
	*/	
	
	for(int k=0; k<kNearest; k++){
		int hitSampleId = d_kNearestHit[sampleId * kNearest + k];
		int missSampleId = d_kNearestMiss[sampleId * kNearest + k];
		
		for(int i=0;i<intsPerInstance;i++){
			int instanceInt = d_packedSampleFeatureMatrix[sampleId * intsPerInstance + i];
			int hitInt = d_packedSampleFeatureMatrix[hitSampleId * intsPerInstance + i];
			int missInt = d_packedSampleFeatureMatrix[missSampleId * intsPerInstance + i];

			for(int offset = 0; offset < 16; offset++)
			{
				int attributeIdx = i * 16 + offset;
				if(d_featureMask[attributeIdx] != true){
					continue;
				}
				
				if(attributeIdx < numOfFeatures)
				{
					int deltaHit = ((instanceInt >> offset * 2) & 0x3) == ((hitInt >> offset * 2) & 0x3)? 0 : 1;
					int deltaMiss = ((instanceInt >> offset * 2) & 0x3) == ((missInt >> offset * 2) & 0x3)? 0 : 1;
					float score = deltaMiss - deltaHit;
					d_weight[sampleId * numOfFeatures + attributeIdx] += score;					
				}
			}
		}
	}

/*		
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
*/
	
	for(int i=0; i<numOfFeatures; i++){						
		atomicAdd(&d_finalWeight[i],d_weight[sampleId * numOfFeatures + i]);
	}
	
}

Result* GpuAcceleratedReliefFBucketSortProcessor::parallelizeCalculationOnStages(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, int* packedSampleFeatureMatrix, bool* featureMask, char* labels){
	
	if(isDebugEnabled()){
		cout<<"numOfSamples="<<numOfSamples<<", numOfFeatures="<<numOfFeatures<<endl;
	}
	
	Timer processing("Processing");
	processing.start();
	
	int kNearest = getKNearest();
	
	bool* d_featureMask;	
	int* d_packedSampleFeatureMatrix;	
	int* d_hitDistanceBuckets;
	int* d_missDistanceBuckets;
	char* d_labels;
		
	//int intsPerInstance = numOfFeatures / 16 + (numOfFeatures % 16 == 0? 0 : 1);
	int intsPerInstance = (int)ceil((float)numOfFeatures / 16);	
	
	cudaMalloc(&d_featureMask, numOfFeatures*sizeof(bool));
	cudaMemcpy(d_featureMask, featureMask, numOfFeatures*sizeof(bool),cudaMemcpyHostToDevice);
	getMemoryInfo("after featureMask cudaMalloc");
	
	cudaMalloc(&d_packedSampleFeatureMatrix, intsPerInstance * numOfSamples*sizeof(int));	
	cudaMemcpy(d_packedSampleFeatureMatrix, packedSampleFeatureMatrix, intsPerInstance * numOfSamples*sizeof(int),cudaMemcpyHostToDevice);	
	getMemoryInfo("after packedSampleFeatureMatrix cudaMalloc");
	
	cudaMalloc(&d_labels, numOfSamples*sizeof(char));
	cudaMemcpy(d_labels, labels, numOfSamples*sizeof(char),cudaMemcpyHostToDevice);
	getMemoryInfo("after labels cudaMalloc");	
	
	
	cudaMalloc(&d_hitDistanceBuckets, numOfSamples * numOfFeatures * (kNearest+1) * sizeof(int));
	cudaMemset(d_hitDistanceBuckets, 0, numOfSamples * numOfFeatures * (kNearest+1) * sizeof(int));
	getMemoryInfo("after hitDistanceBuckets cudaMalloc");
	
	cudaMalloc(&d_missDistanceBuckets, numOfSamples * numOfFeatures * (kNearest+1) * sizeof(int));
	cudaMemset(d_missDistanceBuckets, 0, numOfSamples * numOfFeatures * (kNearest+1) * sizeof(int));
	getMemoryInfo("after missDistanceBuckets cudaMalloc");
	

	int* d_kNearestHit;
	int* d_kNearestMiss;
	cudaMalloc(&d_kNearestHit, kNearest * numOfSamples*sizeof(int));
	getMemoryInfo("after kNearestHit cudaMalloc");
	cudaMalloc(&d_kNearestMiss, kNearest * numOfSamples*sizeof(int));
	getMemoryInfo("after kNearestMiss cudaMalloc");
		
	float* finalWeight = (float*)calloc(numOfFeatures,sizeof(float));
	float* d_weight;
	float* d_finalWeight;
	
	cudaMalloc(&d_weight, numOfSamples*numOfFeatures*sizeof(float));
	cudaMemset(d_weight, 0, numOfSamples*numOfFeatures*sizeof(float));
	getMemoryInfo("after weight cudaMalloc");
	
	
	cudaMalloc(&d_finalWeight, numOfFeatures*sizeof(float));	
	cudaMemset(d_finalWeight, 0, numOfFeatures*sizeof(float));
	getMemoryInfo("after finalWeight cudaMalloc");
		
	int grid2d = (int)ceil(pow(numOfSamples,1/2.));
	int threadSize = getNumberOfThreadsPerBlock();
		
	int samplePerThread = (int)ceil(((float)numOfSamples)/threadSize);	
	
	if(isDebugEnabled()){
		cout<<"grid size="<<grid2d<<"x"<<grid2d<<endl;
		cout<<"thread size="<<threadSize<<endl;
		cout<<"samplePerThread="<<samplePerThread<<endl;
	}
	
	dim3 gridSize(grid2d,grid2d);
	
	if(isDebugEnabled()){
		cout<<"generate distance buckets"<<endl;
	}
	gpu_generateDisatanceBuckets<<<gridSize, threadSize>>>(	
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
	
	if(this->isDebugEnabled()){		
		cout<<"cudaPeekAtLastError:"<<cudaGetErrorString(cudaPeekAtLastError())<<endl;
	}
	
	if(isDebugEnabled()){
		cout<<"weight features"<<endl;
	}	
	
	gpu_weightFeaturesFromBuckets<<<gridSize,1>>>(
		kNearest,
		numOfFeatures,
		numOfSamples,
		intsPerInstance,
		d_kNearestHit,
		d_kNearestMiss,
		d_hitDistanceBuckets,
		d_missDistanceBuckets,
		d_featureMask,
		d_packedSampleFeatureMatrix,
		d_weight,
		d_finalWeight
	);
			
	cudaDeviceSynchronize();
	
	if(this->isDebugEnabled()){		
		cout<<"cudaPeekAtLastError:"<<cudaGetErrorString(cudaPeekAtLastError())<<endl;
	}
	
	cudaMemcpy(finalWeight, d_finalWeight, numOfFeatures*sizeof(float), cudaMemcpyDeviceToHost);
			
	if(isDebugEnabled()){
		cout<<"generate result"<<endl;
	}
	Result* result = new Result;
	result->scores = new double[numOfFeatures];
	int divisor = numOfSamples * kNearest;
	for(int i=0;i<numOfFeatures;i++){
		result->scores[i] = finalWeight[i]/divisor;
	}
	result->success = true;	
		
	free(finalWeight);
	
	cudaFree(d_packedSampleFeatureMatrix);	
	cudaFree(d_labels);
	cudaFree(d_hitDistanceBuckets);
	cudaFree(d_missDistanceBuckets);
	cudaFree(d_featureMask);	
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

