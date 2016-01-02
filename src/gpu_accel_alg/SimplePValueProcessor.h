#ifndef SIMPLEPVALUEPROCESSOR_H
#define SIMPLEPVALUEPROCESSOR_H

#include "SimpleProcessor.h"
class SimplePValueProcessor : public SimpleProcessor 
{
	public:
		
		Result* calculate(int numOfFeatures, 
				char** label0FeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
				char** label1FeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
				bool* featureMask);

};

#endif
