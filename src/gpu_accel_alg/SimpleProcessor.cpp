#include "SimpleProcessor.h"
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include "utils/Timer.h"
#include "threadpool/ThreadPool.h"

using namespace std;

void SimpleProcessor::setNumberOfCores(int numberOfCores){
    this->numberOfCores = numberOfCores;
}

struct AsynCalculateArgs
{
	char *array1;
	int array1_size;
	char *array2;
	int array2_size;	
	int index;
	double *score;
	SimpleProcessor* processor;	
};

void asynCalculateAFeature(void* arg) {	
	AsynCalculateArgs* asynCalculateArgs = (AsynCalculateArgs*) arg;
		
	asynCalculateArgs->processor->calculateAFeature(
		asynCalculateArgs->array1,
		asynCalculateArgs->array1_size,
		asynCalculateArgs->array2,
		asynCalculateArgs->array2_size,
		asynCalculateArgs->score
	);	
	
}

Result* SimpleProcessor::calculate(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels){
	
	if(getParallelizationType() == PARALLELIZE_ON_FEATURES){
		return parallelizeCalculationOnFeatures(numOfSamples, numOfFeatures, sampleFeatureMatrix, featureMask, labels);
	}else if(getParallelizationType() == PARALLELIZE_ON_STAGES){
		return parallelizeCalculationOnStages(numOfSamples, numOfFeatures, sampleFeatureMatrix, featureMask, labels);
	}
	
	return NULL;
}

Result* SimpleProcessor::parallelizeCalculationOnStages(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels){
			
	//group samples by label
	int numOfLabel0Samples = 0;
	int numOfLabel1Samples = 0;
	
	for(int j=0; j<numOfSamples; j++)
	{			
		if((int)labels[j]==0){
			numOfLabel0Samples+=1;		
		}else if((int)labels[j]==1){
			numOfLabel1Samples+=1;
		}
	}
	
	char **label0SamplesArray = (char**)malloc(numOfFeatures * sizeof(char*));
	char **label1SamplesArray = (char**)malloc(numOfFeatures * sizeof(char*));
	for(int arrayId=0; arrayId<numOfFeatures; arrayId++){
		label0SamplesArray[arrayId] = (char*)malloc(numOfLabel0Samples * sizeof(char));
		label1SamplesArray[arrayId] = (char*)malloc(numOfLabel1Samples * sizeof(char));
	}
	
	for(int i=0;i<numOfFeatures;i++){		
		
		if(featureMask[i] != true){
			continue;
		}

		int label0Index=0;
		int label1Index=0;
		
		for(int j=0; j<numOfSamples; j++)
		{			
			int index = j*numOfFeatures + i;
			if(labels[j]==0){
				label0SamplesArray[i][label0Index]=sampleFeatureMatrix[index];
				label0Index+=1;
			}else if(labels[j]==1){
				label1SamplesArray[i][label1Index]=sampleFeatureMatrix[index];
				label1Index+=1;				
			}			
		}			
	}
	
	Result* result = new Result;
	result->scores=new double[numOfFeatures];
		
	calculateAllFeatures(
		label0SamplesArray, numOfLabel0Samples, 
		label1SamplesArray, numOfLabel1Samples, 
		numOfFeatures,
		result->scores,
		&result->success,
		&result->errorMessage);
		
	for(int arrayId=0; arrayId<numOfFeatures; arrayId++){
		free(label0SamplesArray[arrayId]);
		free(label1SamplesArray[arrayId]);
	}
	free(label0SamplesArray);
	free(label1SamplesArray);
	
	return result;
}

Result* SimpleProcessor::parallelizeCalculationOnFeatures(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels){
	
	//group samples by label
	int numOfLabel0Samples = 0;
	int numOfLabel1Samples = 0;
	
	for(int j=0; j<numOfSamples; j++)
	{			
		if((int)labels[j]==0){
			numOfLabel0Samples+=1;		
		}else if((int)labels[j]==1){
			numOfLabel1Samples+=1;
		}
	}
	
	//number of array
	//feature num for CPU 
	int arrayNumbers = numOfFeatures;
			
	int featuresPerArray = getFeaturesPerArray(numOfFeatures, arrayNumbers);	
	
	char **label0SamplesArray_feature = (char**)malloc(numOfFeatures * sizeof(char*));
	char **label1SamplesArray_feature = (char**)malloc(numOfFeatures * sizeof(char*));
	for(int arrayId =0; arrayId<arrayNumbers; arrayId++){
		label0SamplesArray_feature[arrayId] = (char*)malloc(featuresPerArray * numOfLabel0Samples * sizeof(char));
		label1SamplesArray_feature[arrayId] = (char*)malloc(featuresPerArray * numOfLabel1Samples * sizeof(char));
	}
		
	for(int i=0;i<numOfFeatures;i++){
		int arrayId = i / featuresPerArray;
		int featureId = i % featuresPerArray;
		
		//std::cout<<"featuresPerArray="<<featuresPerArray<<", arrayId="<<arrayId<<", featureId="<<featureId<<std::endl;
		
		if(featureMask[i] != true){
			continue;
		}

		int label0Index=0;
		int label1Index=0;
		
		for(int j=0; j<numOfSamples; j++)
		{
			int index = j*numOfFeatures + i;			
			if(labels[j]==0){
				label0SamplesArray_feature[arrayId][featureId * numOfLabel0Samples + label0Index]=sampleFeatureMatrix[index];
				label0Index+=1;
			}else if(labels[j]==1){
				label1SamplesArray_feature[arrayId][featureId * numOfLabel1Samples + label1Index]=sampleFeatureMatrix[index];
				label1Index+=1;				
			}			
		}			
	}
			
	Timer t1("Processing");
	t1.start();		
	
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

	Result* result = new Result;
	result->scores=new double[numOfFeatures];
		
	for(int i=0;i<numOfFeatures;i++){	
		if(featureMask[i] != true){
			result->scores[i]=FEATURE_MASKED;
			continue;
		}
		
		AsynCalculateArgs* asynCalculateArgs = new AsynCalculateArgs;
		
		asynCalculateArgs->array1 = label1SamplesArray_feature[i];
		asynCalculateArgs->array1_size = numOfLabel1Samples;
		asynCalculateArgs->array2 = label0SamplesArray_feature[i];
		asynCalculateArgs->array2_size = numOfLabel0Samples;
		asynCalculateArgs->index = i;
		asynCalculateArgs->score = &result->scores[i];
		asynCalculateArgs->processor = this;
		
		Task* t = new Task(&asynCalculateAFeature, (void*) asynCalculateArgs);
		tp.add_task(t);
	}
	
	tp.waitAll();
	tp.destroy_threadpool();
	
	/*
	for(int i=0;i<numOfFeatures;i++){		
		cout<<"Feature "<<i<<":"<<testResult->scores[i]<<std::endl;		
	}*/
	
	t1.stop();	
	
	t1.printTimeSpent();
	result->success = true;
	result->startTime=t1.getStartTime();
	result->endTime=t1.getStopTime();

	/*
	for(int i=0; i<numOfFeatures;i++){			
		cout<<"final"<<i<<":"<<result->scores[i]<<endl;
	}	
	*/
		
	//free memory
	for(int i=0; i<arrayNumbers; i++) {
		free(label0SamplesArray_feature[i]);
		free(label1SamplesArray_feature[i]);		
	}
	free(label0SamplesArray_feature);
	free(label1SamplesArray_feature);	

	return result;
}

int SimpleProcessor::getNumberOfCores(){
	int numCPU = this->numberOfCores;
	
	if (numCPU > 0){
		return numCPU;
	}else{
		try{
			numCPU = sysconf(_SC_NPROCESSORS_ONLN);
		}catch( const exception& e ) { // reference to the base of a polymorphic object
			cout << e.what() <<endl; // information from length_error printed		
			numCPU = 1;
		}
		return numCPU;
	}
}
