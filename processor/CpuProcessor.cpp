#include "CpuProcessor.h"
#include <stdlib.h>
#include <iostream>

void CpuProcessor::setNumberOfThreads(int numberOfThreads)
{
    this->numberOfThreads = numberOfThreads;
}

Result* CpuProcessor::fastCalculate(char** label0Array, char** label1Array, int label0Size, int label1Size, int numOfSamples, int numOfFeatures, bool* featureMask){
	std::cout << "\nshould be overrided" <<std::endl;
	return new Result;
}

Result* CpuProcessor::calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels){
	
	int label0Size = 0;
	int label1Size = 0;
	
	for(int j=0; j<numOfSamples; j++)
	{			
		if((int)labels[j]==0){
			label0Size+=1;		
		}else if((int)labels[j]==1){
			label1Size+=1;
		}
	}
	
	char** label0Array;
	label0Array = (char**)malloc(sizeof(char*)*numOfFeatures);
	char** label1Array;
	label1Array	= (char**)malloc(sizeof(char*)*numOfFeatures);
	
	for(int i=0;i<numOfFeatures;i++)
	{
		label0Array[i] = (char*)malloc(sizeof(char)*label0Size);
		label1Array[i] = (char*)malloc(sizeof(char)*label1Size);
		if(featureMask[i] != true){
			continue;
		}
		
		int label0Index=0;
		int label1Index=0;
				
		for(int j=0; j<numOfSamples; j++)
		{
			int index = j*numOfFeatures + i;			
			if(labels[j]==0){
				label0Array[i][label0Index]=sampleTimesFeature[index];
				label0Index+=1;
			}else if(labels[j]==1){
				label1Array[i][label1Index]=sampleTimesFeature[index];
				label1Index+=1;
			}
		}		
	}
		
	return fastCalculate(label0Array, label1Array, label0Size, label1Size, numOfSamples, numOfFeatures, featureMask);
}

