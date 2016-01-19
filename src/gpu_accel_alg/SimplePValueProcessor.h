#ifndef SIMPLEPVALUEPROCESSOR_H
#define SIMPLEPVALUEPROCESSOR_H

#include "SimpleProcessor.h"
class SimplePValueProcessor : public SimpleProcessor 
{
	public:
			
		void calculateAFeature(
			char* label0SamplesArray, int numOfLabel0Samples,
			char* label1SamplesArray, int numOfLabel1Samples,
			double* score);

};

#endif
