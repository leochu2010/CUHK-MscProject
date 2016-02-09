#ifndef SIMPLERELIEFFPROCESSOR_H
#define SIMPLERELIEFFPROCESSOR_H

#include "SimpleProcessor.h"
class SimpleReliefFProcessor : public SimpleProcessor 
{
	public:
	
		SimpleReliefFProcessor(int kNearest);
			
		void calculateAllFeatures(
			int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels,
			double* scores, bool* success, string* errorMessage);
	
	private:		
		
		int kNearestInstance;
		
		int getKNearest();
		
};

#endif
