#include "Processor.h"
#include "Timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

void Processor::setDebug(bool debug){
	this->debug = debug;
}
bool Processor::isDebugEnabled(){
	return this->debug;
}

int Processor::getNumberOfFeatureSizeTimesSampleSize2dArrays(int numOfFeatures){
	return 1;
}

int Processor::getFeaturesPerArray(int numOfFeatures, int processingUnitCount){	
	return round((numOfFeatures/(float)processingUnitCount)+0.5f);	
}

Result* Processor::calculate(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels){
		
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
	int arrayNumbers = getNumberOfFeatureSizeTimesSampleSize2dArrays(numOfFeatures);
			
	int featuresPerArray = getFeaturesPerArray(numOfFeatures, arrayNumbers);
	
	char **label0FeatureSizeTimesSampleSize2dArray = (char**)malloc(arrayNumbers * sizeof(char*));
	char **label1FeatureSizeTimesSampleSize2dArray = (char**)malloc(arrayNumbers * sizeof(char*));
	for(int arrayId =0; arrayId<arrayNumbers; arrayId++){
		label0FeatureSizeTimesSampleSize2dArray[arrayId] = (char*)malloc(featuresPerArray * numOfLabel0Samples * sizeof(char));
		label1FeatureSizeTimesSampleSize2dArray[arrayId] = (char*)malloc(featuresPerArray * numOfLabel1Samples * sizeof(char));				
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
				label0FeatureSizeTimesSampleSize2dArray[arrayId][featureId * numOfLabel0Samples + label0Index]=sampleFeatureMatrix[index];				
				label0Index+=1;
			}else if(labels[j]==1){
				label1FeatureSizeTimesSampleSize2dArray[arrayId][featureId * numOfLabel1Samples + label1Index]=sampleFeatureMatrix[index];
				label1Index+=1;				
			}
		}				
	}
		
	Result* result = calculate(numOfFeatures, 
		label0FeatureSizeTimesSampleSize2dArray, numOfLabel0Samples,
		label1FeatureSizeTimesSampleSize2dArray, numOfLabel1Samples, 
		featureMask);
		
	//free memory
	for(int dev=0; dev<arrayNumbers; dev++) {
		free(label0FeatureSizeTimesSampleSize2dArray[dev]);
		free(label1FeatureSizeTimesSampleSize2dArray[dev]);		
	}
	free(label0FeatureSizeTimesSampleSize2dArray);
	free(label1FeatureSizeTimesSampleSize2dArray);	
		
	return result;
}

Result* Processor::calculate(int numOfFeatures, 
		char** label0FeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
		char** label1FeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
		bool* featureMask){
		
		std::cout << "no calculation"<<std::endl;
		
		return 0;
}
