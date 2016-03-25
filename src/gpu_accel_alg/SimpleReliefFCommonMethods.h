#ifndef SIMPLERELIEFFCOMMONMETHODS_H
#define SIMPLERELIEFFCOMMONMETHODS_H

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <pthread.h>
#include <cstdlib>

using namespace std;

extern void pushSampleIdIntoBucket(int sample1Id, int sample2Id, int numOfFeatures, int distance, int kNearest, int* distanceBuckets);
extern void generateDisatanceBuckets(
		int kNearest,
		int numOfSamples,			
		int numOfFeatures,		
		int intsPerInstance,
		char* labels,		
		int* packedSampleFeatureMatrix,		
		int* hitDistanceBuckets,
		int* missDistanceBuckets		
	);
extern void findKNearest(int numOfFeatures, int sampleId, int* distanceBuckets, int* kNearestSampleId, int kNearest);
extern void weightFeatures(
		int kNearest,
		int numOfFeatures,
		int numOfSamples,
		int intsPerInstance,
		int* kNearestHit,
		int* kNearestMiss,
		int* hitDistanceBuckets,
		int* missDistanceBuckets,	
		char* sampleFeatureMatrix,
		int* packedSampleFeatureMatrix,
		bool* featureMask,
		float* weight,
		float* finalWeight
	);

#endif
