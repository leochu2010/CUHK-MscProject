#ifndef SIMPLETTESTPROCESSOR_H
#define SIMPLETTESTEPROCESSOR_H

#include "SimpleProcessor.h"
class SimpleTTestProcessor : public SimpleProcessor 
{
	public:
			
		void calculateAFeature(
			char* label0SamplesArray, int numOfLabel0Samples,
			char* label1SamplesArray, int numOfLabel1Samples,
			double* score);

};

#endif
