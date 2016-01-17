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
	
	//API without preprocessing for processors
		virtual Result* asynCalculate(int numOfFeatures, 
			char** label0SamplesArray_feature, int numOfLabel0Samples,
			char** label1SamplesArray_feature, int numOfLabel1Samples, 			
			bool* featureMask);
	
	protected:
		
		int getNumberOfCores();
		
};

#endif
