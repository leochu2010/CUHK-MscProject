#ifndef PROCESSOR_H
#define PROCESSOR_H

#include "Result.h"



class Processor {	

private:
	bool debug;
	
public:
	//calcalute + preprocessing
	Result* calculate(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels);	
	
	//fast API without preprocessing
	virtual Result* calculate(int numOfFeatures, 
		char** label0FeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
		char** label1FeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
		bool* featureMask);
		
	void setDebug(bool debug);
	
	bool isDebugEnabled();	
	
protected:

	int getFeaturesPerArray(int numOfFeatures, int processingUnitCount);
	
	virtual int getNumberOfFeatureSizeTimesSampleSize2dArrays(int numOfFeatures);	
};

#endif
