#ifndef SIMPLEPROCESSOR_H
#define SIMPLEPROCESSOR_H

#include "Processor.h"

using namespace std;

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
			
		virtual void calculateAllFeatures(
			int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, int* packedSampleFeatureMatrix, bool* featureMask, char* labels,
			double* scores, bool* success, string* errorMessage){};
			
	protected:
	
		Result* parallelizeCalculationOnFeatures(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);
		
		Result* parallelizeCalculationOnStages(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);
	
		int getNumberOfCores();		
		
};

#endif
