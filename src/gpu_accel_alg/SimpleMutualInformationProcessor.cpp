#include "SimpleMutualInformationProcessor.h"
#include "utils/Timer.h"
#include <stdio.h>
#include <math.h>
#include <iostream>

int normaliseArray(char *inputVector, int *outputVector, int vectorLength)
{
  int minVal = 0;
  int maxVal = 0;
  int currentValue;
  int i;
  
  if (vectorLength > 0)
  {
    minVal = (int) floor(inputVector[0]);
    maxVal = (int) floor(inputVector[0]);
  
    for (i = 0; i < vectorLength; i++)
    {
      currentValue = (int) floor(inputVector[i]);
      outputVector[i] = currentValue;
      
      if (currentValue < minVal)
      {
        minVal = currentValue;
      }
      else if (currentValue > maxVal)
      {
        maxVal = currentValue;
      }
    }/*for loop over vector*/
    
    for (i = 0; i < vectorLength; i++)
    {
      outputVector[i] = outputVector[i] - minVal;
    }

    maxVal = (maxVal - minVal) + 1;
  }
  
  return maxVal;
}/*normaliseArray(double*,double*,int)*/



double SimpleMutualInformationProcessor::calculateMutualInformation(char *firstVector, char *secondVector, int vectorLength)
{
  double mutualInformation = 0.0;
  int firstIndex,secondIndex;  
    
  int *firstNormalisedVector;
  int *secondNormalisedVector;
  int *firstStateCounts;
  int *secondStateCounts;
  int *jointStateCounts;
  double *firstStateProbs;
  double *secondStateProbs;
  double *jointStateProbs;
  int firstNumStates;
  int secondNumStates;
  int jointNumStates;
  int i;
  double length = vectorLength;  

  firstNormalisedVector = (int *) calloc(vectorLength,sizeof(int));
  secondNormalisedVector = (int *) calloc(vectorLength,sizeof(int));
  
  firstNumStates = normaliseArray(firstVector,firstNormalisedVector,vectorLength);
  secondNumStates = normaliseArray(secondVector,secondNormalisedVector,vectorLength);
  jointNumStates = firstNumStates * secondNumStates;
  
  firstStateCounts = (int *) calloc(firstNumStates,sizeof(int));
  secondStateCounts = (int *) calloc(secondNumStates,sizeof(int));
  jointStateCounts = (int *) calloc(jointNumStates,sizeof(int));
  
  firstStateProbs = (double *) calloc(firstNumStates,sizeof(double));
  secondStateProbs = (double *) calloc(secondNumStates,sizeof(double));
  jointStateProbs = (double *) calloc(jointNumStates,sizeof(double));
    
  /* optimised version, less numerically stable
  double fractionalState = 1.0 / vectorLength;
  
  for (i = 0; i < vectorLength; i++)
  {
    firstStateProbs[firstNormalisedVector[i]] += fractionalState;
    secondStateProbs[secondNormalisedVector[i]] += fractionalState;
    jointStateProbs[secondNormalisedVector[i] * firstNumStates + firstNormalisedVector[i]] += fractionalState;
  }
  */
  
  /* Optimised for number of FP operations now O(states) instead of O(vectorLength) */
  for (i = 0; i < vectorLength; i++)
  {
    firstStateCounts[firstNormalisedVector[i]] += 1;
    secondStateCounts[secondNormalisedVector[i]] += 1;
    jointStateCounts[secondNormalisedVector[i] * firstNumStates + firstNormalisedVector[i]] += 1;
  }
  
  for (i = 0; i < firstNumStates; i++)
  {
    firstStateProbs[i] = firstStateCounts[i] / length;
  }
  
  for (i = 0; i < secondNumStates; i++)
  {
    secondStateProbs[i] = secondStateCounts[i] / length;
  }
  
  for (i = 0; i < jointNumStates; i++)
  {
    jointStateProbs[i] = jointStateCounts[i] / length;
  }

  free(firstNormalisedVector);
  free(secondNormalisedVector);
  free(firstStateCounts);
  free(secondStateCounts);
  free(jointStateCounts);
    
  firstNormalisedVector = NULL;
  secondNormalisedVector = NULL;
  firstStateCounts = NULL;
  secondStateCounts = NULL;
  jointStateCounts = NULL;
    
  double *jointProbabilityVector;
  int numJointStates;
  double *firstProbabilityVector;
  int numFirstStates;
  double *secondProbabilityVector;
  
	
  jointProbabilityVector = jointStateProbs;
  numJointStates = jointNumStates;
  firstProbabilityVector = firstStateProbs;
  numFirstStates = firstNumStates;
  secondProbabilityVector = secondStateProbs;
  
  /*
  ** I(X;Y) = sum sum p(xy) * log (p(xy)/p(x)p(y))
  */
  for (i = 0; i < numJointStates; i++)
  {
    firstIndex = i % numFirstStates;
    secondIndex = i / numFirstStates;
    
    if ((jointProbabilityVector[i] > 0) && (firstProbabilityVector[firstIndex] > 0) && (secondProbabilityVector[secondIndex] > 0))
    {
      /*double division is probably more stable than multiplying two small numbers together
      ** mutualInformation += jointProbabilityVector[i] * log(jointProbabilityVector[i] / (firstProbabilityVector[firstIndex] * secondProbabilityVector[secondIndex]));
      */
      mutualInformation += jointProbabilityVector[i] * log(jointProbabilityVector[i] / firstProbabilityVector[firstIndex] / secondProbabilityVector[secondIndex]);
    }
  }
  
  //mutualInformation /= log(2.0);
  
  free(firstProbabilityVector);
  firstProbabilityVector = NULL;
  free(secondProbabilityVector);
  secondProbabilityVector = NULL;
  free(jointProbabilityVector);
  jointProbabilityVector = NULL;
  
  return mutualInformation;
}

void SimpleMutualInformationProcessor::calculateAFeature(
	char* label0SamplesArray, int numOfLabel0Samples,
	char* label1SamplesArray, int numOfLabel1Samples,
	double* score){
	
	if (numOfLabel0Samples != numOfLabel1Samples){
		*score=INVALID_FEATURE;
		return;
	}
		
	*score = calculateMutualInformation(label0SamplesArray,label1SamplesArray,numOfLabel0Samples);	
}
