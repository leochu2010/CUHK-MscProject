#include "MutualInformationCpuProcessor.h"
#include "utils/Timer.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "lib/MIToolbox/MIToolbox.h"
#include "lib/MIToolbox/ArrayOperations.h"
#include "lib/MIToolbox/CalculateProbability.h"
#include "lib/MIToolbox/Entropy.h"
#include "lib/MIToolbox/MutualInformation.h"


Result* MutualInformationCpuProcessor::calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels)
{
	Timer t1 ("Total");
	t1.start();
	
	
	Result* testResult = new Result;
	testResult->scores=new double[numOfFeatures];
	
	double labelArray[numOfSamples];	
	for(int i=0;i<numOfSamples;i++)
	{
		labelArray[i]=labels[i];
		//std::cout<<labelArray[i];
	}
	std::cout<<std::endl;	

	int label0Size = 0;
	int label1Size = 0;
	
	for(int j=0; j<numOfSamples; j++)
	{
		if(labels[j]==0){
			label0Size+=1;		
		}else if(labels[j]==1){
			label1Size+=1;
		}
	}
	
	for(int i=0;i<numOfFeatures;i++)
	{
		if(featureMask[i] != true){
			continue;
		}
				
		double label0Array[label0Size];
		double label1Array[label1Size];
		int label0Index=0;
		int label1Index=0;
		
		for(int j=0; j<numOfSamples; j++)
		{
			int index = j*numOfFeatures + i;
			if(labels[j]==0){
				label0Array[label0Index]=sampleTimesFeature[index];
				label0Index+=1;
			}else if(labels[j]==1){
				label1Array[label1Index]=sampleTimesFeature[index];
				label1Index+=1;
			}
		}		
		
				
		double score = this->calculateMutualInformation(label0Array, label1Array, label0Size);		
		testResult->scores[i]=score;
		std::cout<<"Feature "<<i<<":"<<score<<std::endl;
	}
	
	std::cout<<std::endl;
		
	t1.stop();
	t1.printTimeSpent();
	

	testResult->startTime=t1.getStartTime();
	testResult->endTime=t1.getStopTime();
	return testResult;
}

double MutualInformationCpuProcessor::calculateMutualInformation(double *dataVector, double *targetVector, int vectorLength)
{
  double mutualInformation = 0.0;
  int firstIndex,secondIndex;
  int i;
  JointProbabilityState state = calculateJointProbability(dataVector,targetVector,vectorLength);
    
  /*
  ** I(X;Y) = sum sum p(xy) * log (p(xy)/p(x)p(y))
  */
  for (i = 0; i < state.numJointStates; i++)
  {
    firstIndex = i % state.numFirstStates;
    secondIndex = i / state.numFirstStates;
    
    if ((state.jointProbabilityVector[i] > 0) && (state.firstProbabilityVector[firstIndex] > 0) && (state.secondProbabilityVector[secondIndex] > 0))
    {
      /*double division is probably more stable than multiplying two small numbers together
      ** mutualInformation += state.jointProbabilityVector[i] * log(state.jointProbabilityVector[i] / (state.firstProbabilityVector[firstIndex] * state.secondProbabilityVector[secondIndex]));
      */
      mutualInformation += state.jointProbabilityVector[i] * log(state.jointProbabilityVector[i] / state.firstProbabilityVector[firstIndex] / state.secondProbabilityVector[secondIndex]);
    }
  }
  
  //mutualInformation /= log(2.0);
  
  FREE_FUNC(state.firstProbabilityVector);
  state.firstProbabilityVector = NULL;
  FREE_FUNC(state.secondProbabilityVector);
  state.secondProbabilityVector = NULL;
  FREE_FUNC(state.jointProbabilityVector);
  state.jointProbabilityVector = NULL;
  
  return mutualInformation;
}
