#ifndef PROCESSOR_H
#define PROCESSOR_H

#include "Result.h"

class Processor {	

private:
	bool debug;
public:
	virtual Result* calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* label) = 0;
	void setDebug(bool debug);
	bool isDebugEnabled();
};

#endif
