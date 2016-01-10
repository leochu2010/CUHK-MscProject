#ifndef SIMPLEPROCESSOR_H
#define SIMPLEPROCESSOR_H

#include "Processor.h"
class SimpleProcessor : public Processor 
{
	private:
		int numberOfCores;
		
	public:
		void setNumberOfCores(int numberOfCores);
		
		Result* calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);		
		
	protected:
	
		virtual int getNumberOfFeatureSizeTimesSampleSize2dArrays(int numOfFeatures);
		
		int getNumberOfCores();
		
};

#endif
