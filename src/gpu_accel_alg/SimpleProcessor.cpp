#include "SimpleProcessor.h"
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include "utils/Timer.h"

using namespace std;

void SimpleProcessor::setNumberOfCores(int numberOfCores){
    this->numberOfCores = numberOfCores;
}

Result* SimpleProcessor::asynCalculate(int numOfFeatures, 
		char** label0SamplesArray_feature, int numOfLabel0Samples,
		char** label1SamplesArray_feature, int numOfLabel1Samples, 			
		bool* featureMask){
		
	cout << "no calculation"<<endl;
	return new Result;
}

Result* SimpleProcessor::calculate(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels){
		
	Timer pre("Pre-processing");
	pre.start();
	
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
	//device for GPU
	//feature num for CPU 
	int arrayNumbers = numOfFeatures;
			
	int featuresPerArray = getFeaturesPerArray(numOfFeatures, arrayNumbers);	
	
	
	
	char **label0SamplesArray_feature = (char**)malloc(arrayNumbers * sizeof(char*));
	char **label1SamplesArray_feature = (char**)malloc(arrayNumbers * sizeof(char*));
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
		
	Result* result = asynCalculate(numOfFeatures, 
		label0SamplesArray_feature, numOfLabel0Samples,
		label1SamplesArray_feature, numOfLabel1Samples, 		
		featureMask);

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
