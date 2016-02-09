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

struct AsynSampleDistanceArgs
{	
	char* sampleFeatureMatrix;
	int sample1Id;
	int sample2Id;
	int numOfFeatures;
	bool* featureMask;	
	int* distance;
};

void asynSampleDistance(void* arg) {
	AsynSampleDistanceArgs* args = (AsynSampleDistanceArgs*) arg;
			
	char* sampleFeatureMatrix = args->sampleFeatureMatrix;
	int sample1Id = args->sample1Id;
	int sample2Id = args->sample2Id;
	int numOfFeatures = args->numOfFeatures;
	bool* featureMask = args->featureMask;
	int* distance = args->distance;
	
	*distance = 0;
	for(int i=0; i<numOfFeatures; i++){
		if(featureMask[i] != true){
			continue;
		}						
		
		if(sampleFeatureMatrix[sample1Id * numOfFeatures + i] != sampleFeatureMatrix[sample2Id * numOfFeatures + i]){
			*distance += 1;
		}
	}	
}

int** getSampleDisatanceMatrix(
	int numOfSamples, int numOfFeatures, 
	char* sampleFeatureMatrix, bool* featureMask, 
	ThreadPool* tp,
	bool debug){
	
	int** sampleDistanceMatrix = new int*[numOfSamples];
	for(int i=0; i<numOfSamples; i++){
		sampleDistanceMatrix[i] = new int[numOfSamples];
	}
		
	for(int sample1Id=0; sample1Id < numOfSamples; sample1Id++){
		for(int sample2Id=0; sample2Id < numOfSamples; sample2Id++){
			
			AsynSampleDistanceArgs* args = new AsynSampleDistanceArgs;
			args->sampleFeatureMatrix = sampleFeatureMatrix;
			args->sample1Id = sample1Id;
			args->sample2Id = sample2Id;
			args->numOfFeatures = numOfFeatures;
			args->featureMask = featureMask;
			args->distance = &sampleDistanceMatrix[sample1Id][sample2Id];
		
			Task* t = new Task(&asynSampleDistance, (void*) args);
			tp->add_task(t);
		}
	}
	
	if(debug){		
		cout << "numOfSamples:"<<numOfSamples<<endl;
	}
	
	tp->waitAll();	
	return sampleDistanceMatrix;
}

int minDistanceSampleId(int* sampleDistanceMatrix, bool* ignore, int numOfSamples, int numOfFeatures, char *labels, char label, bool sameLabel){
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
		
		if(ignore[i]){
			continue;
		}
		if(sampleDistanceMatrix[i] < minDistance){
			minDistanceIndex = i;
			minDistance = sampleDistanceMatrix[i];
		}
	}
	return minDistanceIndex;
}

struct AsynKNearestArgs{
	int sampleId;
	int numOfSamples;
	int numOfFeatures;
	int* kNearestHit;
	int* kNearestMiss;
	char* labels;
	int* sampleDistanceMatrix;
	int kNearest;
};

void asynNearestSamples(void* arg){
	
	AsynKNearestArgs* args = (AsynKNearestArgs*) arg;
	
	int sampleId = args->sampleId;
	int numOfSamples = args->numOfSamples;
	int* kNearestHit = args->kNearestHit;
	int* kNearestMiss = args->kNearestMiss;
	char* labels = args->labels;
	int* sampleDistanceMatrix = args->sampleDistanceMatrix;
	int kNearest = args->kNearest;
	int numOfFeatures = args->numOfFeatures;
	
	bool* ignoreHit = (bool*)calloc(numOfSamples,sizeof(bool));
	bool* ignoreMiss = (bool*)calloc(numOfSamples,sizeof(bool));
		
	for(int i=0; i<numOfSamples; i++){
		ignoreHit[i]=false;
		ignoreMiss[i]=false;
	}
	
	ignoreHit[sampleId] = true;
	ignoreMiss[sampleId] = true;

	int sameLabel = true;
	for(int i=0; i<kNearest; i++){
		int minDistanceId = minDistanceSampleId(sampleDistanceMatrix, ignoreHit, numOfSamples, numOfFeatures, labels, labels[sampleId], sameLabel);
		ignoreHit[minDistanceId] = true;
		kNearestHit[i] = minDistanceId;
	}
	
	sameLabel = false;
	for(int i=0; i<kNearest; i++){
		int minDistanceId = minDistanceSampleId(sampleDistanceMatrix, ignoreMiss, numOfSamples, numOfFeatures, labels, labels[sampleId], sameLabel);
		ignoreMiss[minDistanceId] = true;
		kNearestMiss[i] = minDistanceId;
	}
	
	free(ignoreHit);
	free(ignoreMiss);
	
}

void kNearestSamples(int **kNearestHit, int **kNearestMiss, int kNearest, int **sampleDistanceMatrix, int numOfSamples, int numOfFeatures, char* labels,
	ThreadPool* tp,
	bool debug){
	
	for(int i=0; i<numOfSamples; i++){		
		
		AsynKNearestArgs* args = new AsynKNearestArgs;
		args->sampleId = i;
		args->numOfSamples = numOfSamples;
		args->numOfFeatures = numOfFeatures;
		args->kNearestHit = kNearestHit[i];
		args->kNearestMiss = kNearestMiss[i];
		args->labels = labels;		
		args->sampleDistanceMatrix = sampleDistanceMatrix[i];				
		args->kNearest = kNearest;
		
		Task* t = new Task(&asynNearestSamples, (void*) args);
		tp->add_task(t);
	}
	tp->waitAll();
	/*
	for(int i=0;i<kNearest;i++){
		cout<<kNearestHit[0][i]<<":"<<sampleDistanceMatrix[0][kNearestHit[0][i]]<<endl;
	}
	*/
	
}

struct AsynWeightArgs{	
	int sampleId;
	int kNearest;
	double* weight;
	char* sampleFeatureMatrix;
	int* kNearestHit;
	int* kNearestMiss;
	int numOfFeatures;
};

void asynWeights(void* arg){
	
	AsynWeightArgs* args = (AsynWeightArgs*)arg;
	
	int sampleId = args->sampleId;
	int kNearest = args->kNearest;
	int numOfFeatures = args->numOfFeatures;
	double* weight = args->weight;
	char* sampleFeatureMatrix = args->sampleFeatureMatrix;
	int* kNearestHit = args->kNearestHit;
	int* kNearestMiss = args->kNearestMiss;
	
	for(int i=0; i<kNearest; i++){
		int hitSampleId = kNearestHit[i];
		int missSampleId = kNearestMiss[i];
		for(int j=0; j<numOfFeatures; j++){
			
			char feature = sampleFeatureMatrix[sampleId * numOfFeatures + j];
			char hitFeature = sampleFeatureMatrix[hitSampleId * numOfFeatures + j];
			char missFeature = sampleFeatureMatrix[missSampleId * numOfFeatures + j];
			
			if (feature != hitFeature){
				weight[j] -= 1;
			}
			
			if (feature != missFeature){
				weight[j] += 1;
			}
		}
	}
}

void weights(double* finalWeight, int numOfFeatures, int numOfSamples,
		char* sampleFeatureMatrix,
		int** kNearestHit, int** kNearestMiss, int kNearest,
		ThreadPool* tp,
		bool debug){		
	
	double** weights = (double**)malloc(numOfSamples * sizeof(double*));
	for(int i=0;i<numOfFeatures; i++){
		weights[i] = (double*)calloc(numOfFeatures, sizeof(double));
	}
	for(int i=0; i<numOfSamples; i++){		
		AsynWeightArgs* args = new AsynWeightArgs;
		args->sampleId = i;
		args->kNearest = kNearest;
		args->weight = weights[i];
		args->sampleFeatureMatrix = sampleFeatureMatrix;
		args->kNearestHit = kNearestHit[i];
		args->kNearestMiss = kNearestMiss[i];
		args->numOfFeatures = numOfFeatures;
		Task* t = new Task(&asynWeights, (void*) args);
		tp->add_task(t);
	}
		
	tp->waitAll();
	
	//sum up	
	for(int i=0; i<numOfSamples; i++){
		for(int j=0; j<numOfFeatures; j++){
			finalWeight[j] +=  weights[i][j];
		}
	}
	
	for(int i=0;i<numOfFeatures; i++){
		free(weights[i]);
	}
	free(weights);
	
	//normize the weights	
	for(int i=0; i<numOfFeatures; i++){
		finalWeight[i]/=(numOfSamples*kNearest);
	}	
	
}

void SimpleReliefFProcessor::calculateAllFeatures(
	int numOfSamples, int numOfFeatures, 
	char* sampleFeatureMatrix, bool* featureMask, char* labels,
	double* scores, bool* success, string* errorMessage){
	
	int cores = getNumberOfCores();
	int poolThreads = 2 * cores;
	if (cores == 1){
		poolThreads = 1;
	}
	
	ThreadPool tp(poolThreads);
	
	int ret = tp.initialize_threadpool();
	if (ret == -1) {
		cerr << "Failed to initialize thread pool!" << endl;
		exit(EXIT_FAILURE);
	}
	
	if(isDebugEnabled()){
		cout<<"calculate distance matrix"<<endl;
	}
	
	int** sampleDistanceMatrix = getSampleDisatanceMatrix(numOfSamples, numOfFeatures, sampleFeatureMatrix, featureMask, &tp, isDebugEnabled());
	
	int kNearest = getKNearest();
	
	int** kNearestHit = (int**)malloc(numOfSamples * sizeof(int*));
	int** kNearestMiss = (int**)malloc(numOfSamples * sizeof(int*));
	for(int i=0; i<numOfSamples; i++){
		kNearestHit[i] = (int*)calloc(kNearest,sizeof(int));	
		kNearestMiss[i] = (int*)calloc(kNearest,sizeof(int));	
	}
	
	if(isDebugEnabled()){
		cout<<"find "<<kNearest<<"-NearestSamples"<<endl;
	}
	kNearestSamples(kNearestHit, kNearestMiss, kNearest, sampleDistanceMatrix, numOfSamples, numOfFeatures, labels, &tp, isDebugEnabled());

	if(isDebugEnabled()){
		cout<<"weight features"<<endl;
	}	
	weights(scores,numOfFeatures, numOfSamples, sampleFeatureMatrix, kNearestHit, kNearestMiss, kNearest, &tp, isDebugEnabled());
	
	for(int i=0; i<numOfSamples; i++){
		free(kNearestHit[i]);
		free(kNearestMiss[i]);
	}
	free(kNearestHit);
	free(kNearestMiss);
	
	tp.destroy_threadpool();
		
	*success = true;
}