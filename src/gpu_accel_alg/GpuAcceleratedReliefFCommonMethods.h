#ifndef GPUACCELERATEDRELIEFFCOMMONMETHODS_H
#define GPUACCELERATEDRELIEFFCOMMONMETHODS_H

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

using namespace std;

extern __device__ void gpu_pushSampleIdIntoBucket(int sample1Id, int sample2Id, int numOfFeatures, int distance, int kNearest, int* d_distanceBuckets);
extern __global__ void gpu_generateDisatanceBuckets(
		int kNearest,
		int samplePerThread,
		int numOfSamples,			
		int numOfFeatures,		
		int intsPerInstance,
		char* d_labels,		
		int* d_packedSampleFeatureMatrix,		
		int* d_hitDistanceBuckets,
		int* d_missDistanceBuckets		
	);
extern __device__ void gpu_findKNearest(int numOfFeatures, int sampleId, int* d_distanceBuckets, int* d_kNearestSampleId, int kNearest);
extern __global__ void gpu_weightFeatures(
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
	);

#endif
