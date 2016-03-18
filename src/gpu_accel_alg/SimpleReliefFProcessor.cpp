#include "SimpleReliefFProcessor.h"
#include "utils/Timer.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include "threadpool/ThreadPool.h"

using namespace std;

SimpleReliefFProcessor::SimpleReliefFProcessor(int kNearest){
	parallelizationType = PARALLELIZE_ON_STAGES;	
	kNearestInstance = kNearest;
}

int SimpleReliefFProcessor::getKNearest(){
	return kNearestInstance;
}

void pushSampleIdIntoBucket(int sample1Id, int sample2Id, int numOfFeatures, int distance, int kNearest, int* distanceBuckets){
		
		int sampleBucketNum = sample1Id * numOfFeatures * (kNearest+1) + (kNearest+1) * distance;
		if(distanceBuckets[sampleBucketNum] < kNearest){		
			distanceBuckets[sampleBucketNum] += 1;	
			int sampleBucketIdx = distanceBuckets[sampleBucketNum];
			distanceBuckets[sampleBucketNum + sampleBucketIdx] = sample2Id;		
		}	
}

void generateDisatanceBuckets(
		int kNearest,
		int numOfSamples,			
		int numOfFeatures,		
		int intsPerInstance,
		char* labels,		
		int* packedSampleFeatureMatrix,		
		int* hitDistanceBuckets,
		int* missDistanceBuckets		
	){
	
	for(int sample1Id =0; sample1Id< numOfSamples; sample1Id++){
				
		for(int sample2Id = 0; sample2Id < numOfSamples; sample2Id++){
									
			int distance = 0;
			for(int k = 0; k < intsPerInstance; k++){
				int first = packedSampleFeatureMatrix[sample1Id * intsPerInstance + k];
				int second = packedSampleFeatureMatrix[sample2Id * intsPerInstance + k];

				int ret = first ^ second;
				for(int l = 0; l < 32; l += 2){
					int diff = (ret >> l) & 3;
					if(diff != 0) distance++; 
				}
			}
			
			if(labels[sample1Id] == labels[sample2Id]){			
				pushSampleIdIntoBucket(sample1Id, sample2Id, numOfFeatures, distance, kNearest, hitDistanceBuckets);
				pushSampleIdIntoBucket(sample2Id, sample1Id, numOfFeatures, distance, kNearest, hitDistanceBuckets);
			}else{			
				pushSampleIdIntoBucket(sample1Id, sample2Id, numOfFeatures, distance, kNearest, missDistanceBuckets);
				pushSampleIdIntoBucket(sample2Id, sample1Id, numOfFeatures, distance, kNearest, missDistanceBuckets);
			}
		}
	}
}

void findKNearest(int numOfFeatures, int sampleId, int* distanceBuckets, int* kNearestSampleId, int kNearest){
	int numOfSamples = 0;	
	for(int distance=0; distance<numOfFeatures; distance++){	
			
		int bucket = sampleId * numOfFeatures * (kNearest+1) + (kNearest+1) * distance;
		
		/*
		if(sampleId == 0){			
			printf("distanceBuckets[%d]=%d, %d, %d, %d, %d, %d \n",distance, distanceBuckets[bucket], 
			distanceBuckets[bucket + 1],
			distanceBuckets[bucket + 2],
			distanceBuckets[bucket + 3],
			distanceBuckets[bucket + 4],
			distanceBuckets[bucket + 5]);
		}
		*/
		
		for(int i=0; i<distanceBuckets[bucket];i++){
			int nearSampleId = distanceBuckets[bucket + i+1];
			kNearestSampleId[sampleId * kNearest + numOfSamples] = nearSampleId;
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

void weightFeatures(
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
	){
		
	for(int sampleId=0; sampleId < numOfSamples; sampleId++){
		
		findKNearest(numOfFeatures, sampleId, hitDistanceBuckets, kNearestHit, kNearest);
		findKNearest(numOfFeatures, sampleId, missDistanceBuckets, kNearestMiss, kNearest);
			
		for(int k=0; k<kNearest; k++){
			int hitSampleId = kNearestHit[sampleId * kNearest + k];
			int missSampleId = kNearestMiss[sampleId * kNearest + k];
			
			for(int i=0;i<intsPerInstance;i++){
				int instanceInt = packedSampleFeatureMatrix[sampleId * intsPerInstance + i];
				int hitInt = packedSampleFeatureMatrix[hitSampleId * intsPerInstance + i];
				int missInt = packedSampleFeatureMatrix[missSampleId * intsPerInstance + i];

				for(int offset = 0; offset < 16; offset++)
				{
					int attributeIdx = i * 16 + offset;
					
					if(featureMask[attributeIdx] != true){
					continue;
				}
					
					if(attributeIdx < numOfFeatures)
					{
						int deltaHit = ((instanceInt >> offset * 2) & 0x3) == ((hitInt >> offset * 2) & 0x3)? 0 : 1;
						int deltaMiss = ((instanceInt >> offset * 2) & 0x3) == ((missInt >> offset * 2) & 0x3)? 0 : 1;
						float score = deltaMiss - deltaHit;
						weight[sampleId * numOfFeatures + attributeIdx] += score;					
					}
				}
			}
			/*
			for(int j=0; j<numOfFeatures; j++){
			
				if(featureMask[j] != true){
					continue;
				}
				
				char feature = sampleFeatureMatrix[sampleId * numOfFeatures + j];
				char hitFeature = sampleFeatureMatrix[hitSampleId * numOfFeatures + j];
				char missFeature = sampleFeatureMatrix[missSampleId * numOfFeatures + j];
				
				if (feature != hitFeature){
					weight[sampleId * numOfFeatures + j] -= 1;
				}
				
				if (feature != missFeature){
					weight[sampleId * numOfFeatures + j] += 1;
				}
			}
			*/
		}

		for(int i=0; i<numOfFeatures; i++){						
			finalWeight[i]+=weight[sampleId * numOfFeatures + i];
		}
	}
	
}
	

void SimpleReliefFProcessor::calculateAllFeatures(
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