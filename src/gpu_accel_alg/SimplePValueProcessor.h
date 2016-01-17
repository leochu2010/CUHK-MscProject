#ifndef SIMPLEPVALUEPROCESSOR_H
#define SIMPLEPVALUEPROCESSOR_H

#include "SimpleProcessor.h"
class SimplePValueProcessor : public SimpleProcessor 
{
	public:
		
		Result* asynCalculate(int numOfFeatures, 
			char** label0SamplesArray_feature, int numOfLabel0Samples,
			char** label1SamplesArray_feature, int numOfLabel1Samples, 			
			bool* featureMask);

};

#endif
