#ifndef SIMPLERELIEFFPROCESSOR_H
#define SIMPLERELIEFFPROCESSOR_H

#include "SimpleProcessor.h"
class SimpleReliefFProcessor : public SimpleProcessor 
{
	public:
	
		SimpleReliefFProcessor();
			
		void calculateAllFeatures(
			char** label0SamplesArray, int numOfLabel0Samples,
			char** label1SamplesArray, int numOfLabel1Samples,
			int numOfFeatures,
			double* scores, bool* success, string* errorMessage);
		
};

#endif
