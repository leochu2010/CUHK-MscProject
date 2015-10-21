#include "MutualInformationCpuProcessor.h"
#include "Timer.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "MIToolbox.h"
#include "ArrayOperations.h"
#include "CalculateProbability.h"
#include "Entropy.h"
#include "MutualInformation.h"


Result* MutualInformationCpuProcessor::calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels)
{
	Timer t1 ("Total");
	t1.start();
	
	
	Result* testResult = new Result;
	testResult->scores=new float[numOfFeatures];
	
	double labelArray[numOfFeatures];	
	for(int i=0;i<numOfFeatures;i++)
	{
		labelArray[i]=labels[i];
		//std::cout<<labelArray[i];
	}
	std::cout<<std::endl;	
	
	for(int i=0;i<numOfFeatures;i++)
	{
		if(featureMask[i] != true){
			continue;
		}
		
						
		double featureArray[numOfSamples];
			
		for(int j=0; j<numOfSamples; j++)
		{
			int index = j*numOfFeatures + i;
			featureArray[j] = sampleTimesFeature[index];
			//std::cout<<featureArray[j];
		}
		
		
		
				
		double score = this->calculateMutualInformation(featureArray, labelArray, numOfSamples);
		//double score = this->calculate_Pvalue(label0Array, label0Size, label1Array, label1Size);
		testResult->scores[i]=score;
		//std::cout<<"Feature "<<i<<":"<<score<<std::endl;	
		//break;	
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
  
  mutualInformation /= log(2.0);
  
  FREE_FUNC(state.firstProbabilityVector);
  state.firstProbabilityVector = NULL;
  FREE_FUNC(state.secondProbabilityVector);
  state.secondProbabilityVector = NULL;
  FREE_FUNC(state.jointProbabilityVector);
  state.jointProbabilityVector = NULL;
  
  return mutualInformation;
}
