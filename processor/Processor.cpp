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

int Processor::getNumberOfProcessingUnit(){
	return 1;
}

int Processor::getFeaturesPerProcessingUnit(int numOfFeatures, int processingUnitCount){	
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
	
	//number of GPU / CPU 
	int processingUnitCount = getNumberOfProcessingUnit();
			
	int featuresPerProcessingUnit = getFeaturesPerProcessingUnit(numOfFeatures, processingUnitCount);
	
	char **label0ProcessingUnitFeatureSizeTimesSampleSize2dArray = (char**)malloc(processingUnitCount * sizeof(char*));
	char **label1ProcessingUnitFeatureSizeTimesSampleSize2dArray = (char**)malloc(processingUnitCount * sizeof(char*));
	for(int processingUnitId =0; processingUnitId<processingUnitCount; processingUnitId++){
		label0ProcessingUnitFeatureSizeTimesSampleSize2dArray[processingUnitId] = (char*)malloc(featuresPerProcessingUnit * numOfLabel0Samples * sizeof(char));
		label1ProcessingUnitFeatureSizeTimesSampleSize2dArray[processingUnitId] = (char*)malloc(featuresPerProcessingUnit * numOfLabel1Samples * sizeof(char));				
	}
	
		
	for(int i=0;i<numOfFeatures;i++){
		int processingUnitId = i / featuresPerProcessingUnit;
		int featureId = i % featuresPerProcessingUnit;
		
		//std::cout<<"featuresPerProcessingUnit="<<featuresPerProcessingUnit<<", processingUnitId="<<processingUnitId<<", featureId="<<featureId<<std::endl;
		
		if(featureMask[i] != true){
			continue;
		}

		int label0Index=0;
		int label1Index=0;
		
		for(int j=0; j<numOfSamples; j++)
		{
			int index = j*numOfFeatures + i;			
			if(labels[j]==0){
				label0ProcessingUnitFeatureSizeTimesSampleSize2dArray[processingUnitId][featureId * numOfLabel0Samples + label0Index]=sampleFeatureMatrix[index];				
				label0Index+=1;
			}else if(labels[j]==1){
				label1ProcessingUnitFeatureSizeTimesSampleSize2dArray[processingUnitId][featureId * numOfLabel1Samples + label1Index]=sampleFeatureMatrix[index];
				label1Index+=1;				
			}
		}				
	}
		
	return calculate(numOfFeatures, 
		label0ProcessingUnitFeatureSizeTimesSampleSize2dArray, numOfLabel0Samples,
		label1ProcessingUnitFeatureSizeTimesSampleSize2dArray, numOfLabel1Samples, 
		featureMask);
}

Result* Processor::calculate(int numOfFeatures, 
		char** label0ProcessingUnitFeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
		char** label1ProcessingUnitFeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
		bool* featureMask){
		
		std::cout << "no calculation"<<std::endl;
		
		return 0;
}
