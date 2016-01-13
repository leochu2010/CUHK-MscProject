#ifndef PROCESSOR_H
#define PROCESSOR_H

#include "Result.h"



class Processor {	

private:
	bool debug;
	
public:
	//calcalute + preprocessing
	Result* calculate(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels);	
	
	//API without preprocessing for processors
	virtual Result* calculate(int numOfFeatures, 
		char** label0FeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
		char** label1FeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
		int* numOfFeaturesPerArray,
		bool* featureMask);
		
	void setDebug(bool debug);
	
	bool isDebugEnabled();	
	
protected:

	int getFeaturesPerArray(int numOfFeatures, int processingUnitCount);
	
	virtual int getNumberOfFeatureSizeTimesSampleSize2dArrays(int numOfFeatures);	
};

#endif
