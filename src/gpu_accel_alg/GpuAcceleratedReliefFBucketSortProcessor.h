#ifndef GPUACCELERATEDRELIEFFBUCKETSORTPROCESSOR_H
#define GPUACCELERATEDRELIEFFBUCKETSORTPROCESSOR_H

#include "GpuAcceleratedProcessor.h"
#include <string>

using namespace std;

class GpuAcceleratedReliefFBucketSortProcessor : public GpuAcceleratedProcessor 
{
	public:

		GpuAcceleratedReliefFBucketSortProcessor(int kNearest);
		
		Result* parallelizeCalculationOnStages(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, int* packedSampleFeatureMatrix, bool* featureMask, char* labels);
	
	private:		
		
		int kNearestInstance;
		
		int getKNearest();

};

#endif
