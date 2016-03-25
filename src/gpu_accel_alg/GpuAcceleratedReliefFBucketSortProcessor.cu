#include "GpuAcceleratedReliefFBucketSortProcessor.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "utils/Timer.h"
#include "GpuAcceleratedReliefFCommonMethods.h"

using namespace std;

GpuAcceleratedReliefFBucketSortProcessor::GpuAcceleratedReliefFBucketSortProcessor(int kNearest){
	parallelizationType = PARALLELIZE_ON_STAGES;
	kNearestInstance = kNearest;
}

int GpuAcceleratedReliefFBucketSortProcessor::getKNearest(){
	return kNearestInstance;
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
	
	gpu_weightFeatures<<<gridSize,1>>>(
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
	int divider = numOfSamples * kNearest;
	for(int i=0;i<numOfFeatures;i++){
		result->scores[i] = finalWeight[i]/divider;
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

