#ifndef PROCESSOR_H
#define PROCESSOR_H

#include "Result.h"

#define FEATURE_MASKED 2

class Processor {	

private:
	bool debug;
	
public:
	//calcalute Interface
	virtual Result* calculate(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels){
		return new Result;
	};	
			
	void setDebug(bool debug);
	
	bool isDebugEnabled();	

protected:
	int getFeaturesPerArray(int numOfFeatures, int arrayNumbers);
	
};

#endif
