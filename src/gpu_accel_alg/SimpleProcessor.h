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
					
		virtual void calculateAFeature(
			char* label0SamplesArray, int numOfLabel0Samples,
			char* label1SamplesArray, int numOfLabel1Samples,
			double* score){};
			
	protected:
	
		int getNumberOfCores();		
		
};

#endif
