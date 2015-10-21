#ifndef MIUTUALINFORMATIONCPUPROCESSOR_H
#define MIUTUALINFORMATIONCPUPROCESSOR_H

#include "CpuProcessor.h"
class MutualInformationCpuProcessor : public CpuProcessor 
{
	private:
			double calculateMutualInformation(double *dataVector, double *targetVector, int vectorLength);			
	public:
        	virtual Result* calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);

};

#endif
