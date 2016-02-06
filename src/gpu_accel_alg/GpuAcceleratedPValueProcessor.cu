#include "GpuAcceleratedPValueProcessor.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

using namespace std;

GpuAcceleratedPValueProcessor::GpuAcceleratedPValueProcessor(){
	parallelizationType = PARALLELIZE_ON_FEATURES;
}

//declare local function
__global__ void calculate_Pvalue(
	char *d_array1, size_t array1_size, size_t array1_size_per_thread,
	char *d_array2, size_t array2_size, size_t array2_size_per_thread, 
	double *d_score,
	bool *d_featureMask,
	size_t numOfFeatures,
	size_t NLoopPerThread,
	int device);

void GpuAcceleratedPValueProcessor::calculateOnStream(int* numberOfFeaturesPerStream,
	char** label0SamplesArray_stream_feature, int numOfLabel0Samples,
	char** label1SamplesArray_stream_feature, int numOfLabel1Samples,
	bool** featureMasksArray_stream_feature,
	double** score,
	int device,
	cudaStream_t* streams,
	bool* success, string* errorMessage){
	
	*success = true;
	
	/*
	if(device == 1){
		cout<<"[GpuAcceleratedPValueProcessor]"<<"numberOfFeaturesPerStream[0]="<<numberOfFeaturesPerStream[0]<<", label0SamplesArray_stream_feature[0][0]="<<0+label0SamplesArray_stream_feature[0][0]<<endl;	
	}
	*/
	
			
	int threadSize = getNumberOfThreadsPerBlock();
	size_t label0SizePerThread = ceil((numOfLabel0Samples/(float)(threadSize)));
	size_t label1SizePerThread = ceil((numOfLabel1Samples/(float)(threadSize)));
	
	//copy data from main memory to GPU	
	int streamCount = getNumberOfStreamsPerDevice();
	char *d_label0Array[streamCount];
	char *d_label1Array[streamCount];
	double *d_score[streamCount];
	bool *d_featureMask[streamCount];
	
	if(this->isDebugEnabled()){
		cout << "copy to GPU"<<endl;
	}
		
	cudaSetDevice(device);
	
	int maxFeaturesPerStream = numberOfFeaturesPerStream[0];

	for(int i=0; i<streamCount; i++){
		cudaMalloc(&d_label0Array[i],maxFeaturesPerStream*numOfLabel0Samples*sizeof(char));
		cudaMalloc(&d_label1Array[i],maxFeaturesPerStream*numOfLabel1Samples*sizeof(char));
		cudaMalloc(&d_score[i],maxFeaturesPerStream*sizeof(double));
		cudaMalloc(&d_featureMask[i],maxFeaturesPerStream*sizeof(bool));
	}	
	
	for(int i=0; i<streamCount; i++){
		int features = numberOfFeaturesPerStream[i];
		cudaMemcpyAsync(d_label0Array[i],label0SamplesArray_stream_feature[i],features*numOfLabel0Samples*sizeof(char),cudaMemcpyHostToDevice,streams[i]);
		cudaMemcpyAsync(d_label1Array[i],label1SamplesArray_stream_feature[i],features*numOfLabel1Samples*sizeof(char),cudaMemcpyHostToDevice,streams[i]);
		cudaMemcpyAsync(d_score[i],score[i],features*sizeof(double),cudaMemcpyHostToDevice,streams[i]);
		cudaMemcpyAsync(d_featureMask[i],featureMasksArray_stream_feature[i],features*sizeof(bool),cudaMemcpyHostToDevice,streams[i]);
	}
	
	const size_t N = 65535;
	size_t NLoopPerThread = ceil(((float)N)/threadSize);
		
	int grid2d = (int)ceil(pow(maxFeaturesPerStream,1/2.));	
	if(this->isDebugEnabled()){
		cout<<"maxFeaturesPerStream="<<maxFeaturesPerStream<<",grid2d="<<grid2d<<",numOfLabel0Samples="<<numOfLabel0Samples<<",numOfLabel1Samples="<<numOfLabel1Samples<<endl;
	}
	
	dim3 gridSize(grid2d,grid2d);
	
	//calculate	
	for(int i=0; i<streamCount; i++){
		calculate_Pvalue<<<gridSize, threadSize, 0, streams[i]>>>(
				d_label1Array[i], numOfLabel1Samples, label1SizePerThread, 
				d_label0Array[i], numOfLabel0Samples, label0SizePerThread, 
				d_score[i], 
				d_featureMask[i],
				numberOfFeaturesPerStream[i],
				NLoopPerThread,
				device);
	}
			
	if(this->isDebugEnabled()){
		cout<<"cudaPeekAtLastError:"<<cudaPeekAtLastError()<<endl;
	}	
		
	//copy result from GPU to main memory
	for(int i=0; i<streamCount; i++){
		int features = numberOfFeaturesPerStream[i];
		cudaMemcpyAsync(score[i], d_score[i], features*sizeof(double), cudaMemcpyDeviceToHost,streams[i]);
	}

	cudaDeviceSynchronize();
	
	//free cuda memory	
	//destroy streams
	for(int i=0; i<streamCount; i++){
		if(isDebugEnabled()){
			cout<<"cudafree resources"<<endl;
		}	
		cudaFree(d_label1Array[i]);
		cudaFree(d_label0Array[i]);
		cudaFree(d_score[i]);
		cudaFree(d_featureMask[i]);		
	}	
}	

__global__ void calculate_Pvalue(
	char *d_array1, size_t array1_size, size_t array1_size_per_thread,
	char *d_array2, size_t array2_size, size_t array2_size_per_thread, 
	double *d_score,	
	bool *d_featureMask,
	size_t numOfFeatures,
	size_t NLoopPerThread,
	int device) {
		
	//gridDim.x = gridDim.y as it is 2d 
	//idx is feature index of a block on this device
	int idx = gridDim.x * blockIdx.x + blockIdx.y;	
	
	__shared__ float mean1;
	__shared__ float mean2;
	__shared__ float variance1;
	__shared__ float variance2;
		
	if (d_featureMask[idx] != true){			
		return;
	}
		
	if(threadIdx.x == 0){		
		mean1=0;
		mean2=0;
		variance1=0;
		variance2=0;
	}
	
	__syncthreads();
	
	if (idx < numOfFeatures){
		
		//printf("idx=%d, pitch0=%d, pitch1=%d \n", idx, pitch0, pitch1);			
				
		if(threadIdx.x == 0){
			if (array1_size <= 1) {
				d_score[idx] = 1.0;			
				return;
			}
			
			if (array2_size <= 1) {
				d_score[idx] = 1.0;
				return;
			}
		}
		
		if(array1_size_per_thread*(threadIdx.x)< array1_size){
			int m1=0;
			for(int i=array1_size_per_thread*(threadIdx.x); i<array1_size_per_thread*(threadIdx.x+1) && i<array1_size; i++){				
				m1+=d_array1[array1_size * idx + i];
				
			/*
				if(idx==0 && dev==0){
					printf("i1=%d, value=%d\n",i,array1[i]);
				}
			*/
							
			}
			
			/*
			if(idx==0){
				printf("\n threadIdx.x=%d, m1=%d",threadIdx.x, m1);
			}
			*/
			atomicAdd(&mean1,m1);
		}
					
		if(array2_size_per_thread*(threadIdx.x)< array2_size){
			int m2=0;
			for(int i=array2_size_per_thread*(threadIdx.x); i<array2_size_per_thread*(threadIdx.x+1) && i<array2_size; i++){
				m2+=d_array2[array2_size * idx + i];
				
				/*
				if(idx==1 && dev==0){
					printf("i2=%d, value=%d\n",i,d_array2[array2_size * idx + i]);
				}
				*/
				
			}
			atomicAdd(&mean2,m2);
		}
		
		__syncthreads();
		/*
		if(threadIdx.x == 0 && idx==0){
			printf("\n mean1=%f\n",mean1);
		}
		*/
		
		if(threadIdx.x == 0){
			if (mean1 == mean2) {			
				d_score[idx] = 1.0;
				return;
			}

			mean1 /= array1_size;
			mean2 /= array2_size;				
		}
				
		__syncthreads();
		
		/*
		if(threadIdx.x == 0 && idx==0 && device == 1){
			printf("\n device=%d, mean1=%f, mean2=%f",device ,mean1, mean2);			
		}
		*/
		
		if(array1_size_per_thread*(threadIdx.x) < array1_size){
			float v1 = 0;
			float v1s = 0;
			for(int i=array1_size_per_thread*(threadIdx.x); i<array1_size_per_thread*(threadIdx.x +1) && i<array1_size; i++){
				v1=(mean1-d_array1[array1_size * idx + i]);
				v1s += v1*v1; 
			}			
			atomicAdd(&variance1, v1s);
		}			

		if(array2_size_per_thread*(threadIdx.x) < array2_size){
			float v2 = 0;
			float v2s = 0;
			for(int i=array2_size_per_thread*(threadIdx.x); i<array2_size_per_thread*(threadIdx.x+1) && i<array2_size; i++){
				v2=(mean2-d_array2[array2_size * idx + i]);
				v2s += v2*v2;
			}	
			atomicAdd(&variance2, v2s);
		}		
				
		__syncthreads();
		if (threadIdx.x == 0){
			if ((variance1 == 0.0) && (variance2 == 0.0)) {
				d_score[idx] = 1.0;
				return;		
			}				
		}
		
		if(threadIdx.x == 0){
			variance1 = variance1/(array1_size-1);
			variance2 = variance2/(array2_size-1);
		}
		__syncthreads();
		
		
		/*
		if(threadIdx.x == 0 && idx==0 && device == 1){
			printf("\n device=%d, variance1=%f, variance2=%f",device ,variance1, variance2);			
		}
		*/		
		
		__shared__ float sum1;
		__shared__ float sum2;
		__shared__ double h;
		__shared__ double a;
		__shared__ double x;
		
		if (threadIdx.x == 0){
			
			const double WELCH_T_STATISTIC = (mean1-mean2)/sqrt(variance1/array1_size+variance2/array2_size);
			const double DEGREES_OF_FREEDOM = pow((double)(variance1/array1_size+variance2/array2_size),2.0)//numerator
			 /
			(
				(variance1*variance1)/(array1_size*array1_size*(array1_size-1))+
				(variance2*variance2)/(array2_size*array2_size*(array2_size-1))
			);

			a = DEGREES_OF_FREEDOM/2, x = DEGREES_OF_FREEDOM/(WELCH_T_STATISTIC*WELCH_T_STATISTIC+DEGREES_OF_FREEDOM);			
			h = x/65535;
			sum1=0;
			sum2=0;
		}
		__syncthreads();
		
		
		float s1=0;
		float s2=0;
		for(unsigned int i = (threadIdx.x)*(NLoopPerThread); i < (threadIdx.x+1)*(NLoopPerThread); i++) {
			if(i<65535){					
				s1 += (pow(h * i + h / 2.0,a-1))/(sqrt(1-(h * i + h / 2.0)));
				s2 += (pow(h * i,a-1))/(sqrt(1-h * i));
			}
		}
		atomicAdd(&sum1, s1);
		atomicAdd(&sum2, s2);		

		__syncthreads();
		
		/*
		if(device == 1 && threadIdx.x == 0 && idx==0){
			printf("device=%d, idx=%d: sum1=%f, sum2=%f, NLoopPerThread=%d\n",device, idx,sum1,sum2,NLoopPerThread);			
		}
		*/		
		
		if (threadIdx.x == 0){
			double return_value = ((h / 6.0) * ((pow(x,a-1))/(sqrt(1-x)) + 4.0 * sum1 + 2.0 * sum2))/(exp(lgamma(a)+0.57236494292470009-lgamma(a+0.5)));			
			if ((isfinite(return_value) == 0) || (return_value > 1.0)) {
				d_score[idx] = 1.0;		
			} else {							
				d_score[idx] = return_value;
				
			/*
			if ((idx==1 || idx==0)&& device == 1){
				printf("idx=%d: sum1=%f, sum2=%f\n",idx,sum1,sum2);

				printf("idx=%d: T-Test=%f, score=%f\n",idx,((mean1-mean2)/sqrt(variance1/array1_size+variance2/array2_size)),return_value);
			
				printf("idx=%d: mean1=%f,mean2=%f,variance1=%f,array1_size=%d,variance2=%f,array2_size=%d, T-Test=%f\n",idx,mean1,mean2,variance1,array1_size,variance2,array2_size,((mean1-mean2)/sqrt(variance1/array1_size+variance2/array2_size)),return_value);
			}
			*/		
				
			}
		}
		
	}	
}
