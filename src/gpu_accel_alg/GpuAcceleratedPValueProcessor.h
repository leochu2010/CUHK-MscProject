#ifndef GPUACCELERATEDPVALUEGPUPROCESSOR_H
#define GPUACCELERATEDPVALUEGPUPROCESSOR_H

#include "GpuAcceleratedProcessor.h"
class GpuAcceleratedPValueProcessor : public GpuAcceleratedProcessor 
{
	public:
	
		void calculateOnStream(int* numberOfFeaturesPerStream,
			char** label0SamplesArray_stream_feature, int numOfLabel0Samples,
			char** label1SamplesArray_stream_feature, int numOfLabel1Samples,
			bool* featureMask,		
			double** score,
			int device,
			cudaStream_t* streams);

};

#endif
