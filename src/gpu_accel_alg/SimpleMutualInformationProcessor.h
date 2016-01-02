#ifndef SIMPLEMIUTUALINFORMATIONPROCESSOR_H
#define SIMPLEMIUTUALINFORMATIONPROCESSOR_H

#include "SimpleProcessor.h"
class SimpleMutualInformationProcessor : public SimpleProcessor 
{
	private:
			double calculateMutualInformation(double *dataVector, double *targetVector, int vectorLength);			
	public:
        	virtual Result* calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);

};

#endif
