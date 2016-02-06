#ifndef SIMPLEMIUTUALINFORMATIONPROCESSOR_H
#define SIMPLEMIUTUALINFORMATIONPROCESSOR_H

#include "SimpleProcessor.h"
class SimpleMutualInformationProcessor : public SimpleProcessor 
{
	private:			
		double calculateMutualInformation(char *firstVector, char *secondVector, int vectorLength);
	public:        				
		
		SimpleMutualInformationProcessor();
	
		void calculateAFeature(
			char* label0SamplesArray, int numOfLabel0Samples,
			char* label1SamplesArray, int numOfLabel1Samples,
			double* score);

};

#endif
