#include "SimpleReliefFBucketSortProcessor.h"
#include "utils/Timer.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include "threadpool/ThreadPool.h"
#include "SimpleReliefFCommonMethods.h"

using namespace std;

SimpleReliefFBucketSortProcessor::SimpleReliefFBucketSortProcessor(int kNearest){
	parallelizationType = PARALLELIZE_ON_STAGES;	
	kNearestInstance = kNearest;
}

int SimpleReliefFBucketSortProcessor::getKNearest(){
	return kNearestInstance;
}

void SimpleReliefFBucketSortProcessor::calculateAllFeatures(
	int numOfSamples, int numOfFeatures, 
	char* sampleFeatureMatrix, int* packedSampleFeatureMatrix,
	bool* featureMask, char* labels,
	double* scores, bool* success, string* errorMessage){
	
	int kNearest = getKNearest();
	int intsPerInstance = (int)ceil((float)numOfFeatures / 16);	
	
	int* hitDistanceBuckets = (int*)calloc(numOfSamples * numOfFeatures * (kNearest+1),sizeof(int));	
	int* missDistanceBuckets  = (int*)calloc(numOfSamples * numOfFeatures * (kNearest+1),sizeof(int));		
		
	int* kNearestHit = (int*)calloc(kNearest * numOfSamples, sizeof(int));
	int* kNearestMiss = (int*)calloc(kNearest * numOfSamples, sizeof(int));	
	
	float* finalWeight = (float*)calloc(numOfFeatures,sizeof(float));
	float* weight = (float*)calloc(numOfSamples*numOfFeatures,sizeof(float));	
	
	if(isDebugEnabled()){
		cout<<"generate distance buckets"<<endl;
	}
	
	generateDisatanceBuckets(
		kNearest,
		numOfSamples,			
		numOfFeatures,		
		intsPerInstance,
		labels,		
		packedSampleFeatureMatrix,		
		hitDistanceBuckets,
		missDistanceBuckets		
	);
	
	weightFeatures(
		kNearest,
		numOfFeatures,
		numOfSamples,
		intsPerInstance,
		kNearestHit,
		kNearestMiss,
		hitDistanceBuckets,
		missDistanceBuckets,		
		sampleFeatureMatrix,
		packedSampleFeatureMatrix,
		featureMask,
		weight,
		finalWeight
	);
	
	int divider = numOfSamples * kNearest;
	for(int i=0;i<numOfFeatures;i++){
		scores[i] = finalWeight[i]/divider;
	}
	
	free(hitDistanceBuckets);
	free(missDistanceBuckets);
	free(kNearestHit);
	free(kNearestMiss);
	free(weight);
	free(finalWeight);
		
	*success = true;
}