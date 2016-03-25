#ifndef SIMPLERELIEFFBUCKETSORTPROCESSOR_H
#define SIMPLERELIEFFBUCKETSORTPROCESSOR_H

#include "SimpleProcessor.h"
class SimpleReliefFBucketSortProcessor : public SimpleProcessor 
{
	public:
	
		SimpleReliefFBucketSortProcessor(int kNearest);
			
		void calculateAllFeatures(
			int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, int* packedSampleFeatureMatrix, bool* featureMask, char* labels,
			double* scores, bool* success, string* errorMessage);
	
	private:		
		
		int kNearestInstance;
		
		int getKNearest();
		
};

#endif
