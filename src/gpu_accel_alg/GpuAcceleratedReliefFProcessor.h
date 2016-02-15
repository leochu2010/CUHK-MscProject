#ifndef GPUACCELERATEDRELIEFFPROCESSOR_H
#define GPUACCELERATEDRELIEFFPROCESSOR_H

#include "GpuAcceleratedProcessor.h"
#include <string>

using namespace std;

class GpuAcceleratedReliefFProcessor : public GpuAcceleratedProcessor 
{
	public:

		GpuAcceleratedReliefFProcessor(int kNearest);
		
		Result* parallelizeCalculationOnStages(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask, char* labels);
	
	private:		
		
		int kNearestInstance;
		
		int getKNearest();

};

#endif
