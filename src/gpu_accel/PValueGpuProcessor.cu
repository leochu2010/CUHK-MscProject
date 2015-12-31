#include "PValueGpuProcessor.h"
#include "Timer.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

//declare local function
__global__ void calculate_Pvalue(
	char *d_array1, size_t array1_size, size_t array1_size_per_thread,
	char *d_array2, size_t array2_size, size_t array2_size_per_thread, 
	double *d_score, 
	size_t dev, size_t featuresPerDevice, 
	size_t numOfFeatures,
	size_t NLoopPerThread);
	
//implement API	
Result* PValueGpuProcessor::calculate(int numOfFeatures, 
	char** label0FeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
	char** label1FeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
	bool* featureMask){
			
	//get device and thread numbers	
	int deviceCount = getNumberOfDevice();	
	int featuresPerDevice = getFeaturesPerArray(numOfFeatures, deviceCount);
			
	double score[deviceCount][featuresPerDevice];
		
	int threadSize = getNumberOfThreadsPerBlock();
	size_t label0SizePerThread = round((numOfLabel0Samples/(float)(threadSize))+0.5f);
	size_t label1SizePerThread = round((numOfLabel1Samples/(float)(threadSize))+0.5f);
		
	cudaProfilerStart();
	Timer totalProcessing("Total Processing");
	totalProcessing.start();
	
	
	//copy data from main memory to GPU	
	char *d_label0Array[deviceCount];
	char *d_label1Array[deviceCount];
	double *d_score[deviceCount];
		
	if(this->isDebugEnabled()){
		std::cout << "copy to GPU"<<std::endl;
	}
	for(int dev=0; dev<deviceCount; dev++) {
		cudaSetDevice(dev);
				
		cudaMalloc(&d_label0Array[dev],featuresPerDevice*numOfLabel0Samples*sizeof(char));
		cudaMemcpy(d_label0Array[dev],label0FeatureSizeTimesSampleSize2dArray[dev],featuresPerDevice*numOfLabel0Samples*sizeof(char),cudaMemcpyHostToDevice);
		
		cudaMalloc(&d_label1Array[dev],featuresPerDevice*numOfLabel1Samples*sizeof(char));
		cudaMemcpy(d_label1Array[dev],label1FeatureSizeTimesSampleSize2dArray[dev],featuresPerDevice*numOfLabel1Samples*sizeof(char),cudaMemcpyHostToDevice);
				
		cudaMalloc(&d_score[dev],featuresPerDevice*sizeof(double));		
		cudaMemcpy(d_score[dev],score[dev],featuresPerDevice*sizeof(double),cudaMemcpyHostToDevice);		
	}	
	
	int grid2d = (int)round(pow(featuresPerDevice,1/2.)+0.5f);
	if(this->isDebugEnabled()){
		std::cout<<"featuresPerDevice="<<featuresPerDevice<<",grid2d="<<grid2d<<",label0SizePerThread="<<label0SizePerThread<<",label1SizePerThread="<<label1SizePerThread<<std::endl;
	}
	
	dim3 gridSize(grid2d,grid2d);
	
	const size_t N = 65535;
	size_t NLoopPerThread = round(((float)N)/threadSize+0.5f);		
	
	//calculate	
	for(size_t dev=0; dev<deviceCount; dev++) {
		cudaSetDevice(dev);
		calculate_Pvalue<<<gridSize, threadSize>>>(
			d_label1Array[dev], numOfLabel1Samples, label1SizePerThread, 
			d_label0Array[dev], numOfLabel0Samples, label0SizePerThread, 
			d_score[dev], 
			dev, featuresPerDevice, 
			numOfFeatures,
			NLoopPerThread);
	}
		
	if(this->isDebugEnabled()){
		std::cout<<"cudaPeekAtLastError:"<<cudaPeekAtLastError()<<std::endl;
	}
	cudaDeviceSynchronize();
	
	//copy result from GPU to main memory
	for(int dev=0; dev<deviceCount; dev++) {
		cudaSetDevice(dev);
		cudaMemcpy(score[dev], d_score[dev], featuresPerDevice*sizeof(double), cudaMemcpyDeviceToHost);
	}	
	
	//free cuda memory
	for(int dev=0; dev<deviceCount; dev++) {
		cudaFree(d_label1Array[dev]);
		cudaFree(d_label0Array[dev]);
		cudaFree(d_score[dev]);
	}
	cudaFree(d_label1Array);
	cudaFree(d_label0Array);
	cudaFree(d_score);
	
	cudaProfilerStop();
	
	//return result
	Result* testResult = new Result;
	testResult->scores=new double[numOfFeatures];
	for(int dev=0; dev<deviceCount; dev++) {		
		for(int i=0; i<featuresPerDevice;i++){
			int featureId = dev*featuresPerDevice+i;
			if(featureId<numOfFeatures){
				testResult->scores[featureId]=score[dev][i];
				//std::cout<<"Feature "<<featureId<<":"<<score[dev][i]<<std::endl;		
			}
		}	
	}
	
	if(this->isDebugEnabled()){
		std::cout<<std::endl;
	}
		
	totalProcessing.stop();	
	testResult->startTime=totalProcessing.getStartTime();
	testResult->endTime=totalProcessing.getStopTime();		
	testResult->success=true;
		
	
	return testResult;	
}


__global__ void calculate_Pvalue(
	char *d_array1, size_t array1_size, size_t array1_size_per_thread,
	char *d_array2, size_t array2_size, size_t array2_size_per_thread, 
	double *d_score, 
	size_t dev, size_t featuresPerDevice, 
	size_t numOfFeatures,
	size_t NLoopPerThread) {
		
	//gridDim.x = gridDim.y as it is 2d 
	//idx is feature of a block of this device
	int idx = gridDim.x * blockIdx.x + blockIdx.y;	
	
	__shared__ float mean1;
	__shared__ float mean2;
	__shared__ float variance1;
	__shared__ float variance2;
		
	if(threadIdx.x == 0){
		mean1=0;
		mean2=0;
		variance1=0;
		variance2=0;
	}
	
	__syncthreads();
	
	if (idx < featuresPerDevice && dev*featuresPerDevice+idx < numOfFeatures){
		//if (dev==0){
			//printf("dev=%d, idx=%d, pitch0=%d, pitch1=%d \n",dev, idx, pitch0, pitch1);	
		//}		
				
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
		if(threadIdx.x == 0 && idx==0){
			printf("\n mean1=%f, mean2=%f",mean1, mean2);			
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
		if(threadIdx.x == 0 && idx==0){
			printf("\n variance1=%f, variance2=%f",variance1, variance2);			
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
		if(threadIdx.x == 0 && idx==0){
			printf("idx=%d: sum1=%f, sum2=%f, NLoopPerThread=%d, threads=%d\n",idx,sum1,sum2,NLoopPerThread, threads);			
		}
		*/
		
		
		if (threadIdx.x == 0){
			double return_value = ((h / 6.0) * ((pow(x,a-1))/(sqrt(1-x)) + 4.0 * sum1 + 2.0 * sum2))/(exp(lgamma(a)+0.57236494292470009-lgamma(a+0.5)));			
			if ((isfinite(return_value) == 0) || (return_value > 1.0)) {
				d_score[idx] = 1.0;		
			} else {							
				d_score[idx] = return_value;
				
				/*
				if (dev==0 && idx==1){
					//printf("idx=%d: sum1=%f, sum2=%f\n",idx,sum1,sum2);
					//printf("idx=%d: T-Test=%f, score=%f\n",dev*featuresPerDevice+idx,((mean1-mean2)/sqrt(variance1/array1_size+variance2/array2_size)),return_value);
					//printf("idx=%d: mean1=%f,mean2=%f,variance1=%f,array1_size=%d,variance2=%f,array2_size=%d, T-Test=%f\n",dev*featuresPerDevice+idx,mean1,mean2,variance1,array1_size,variance2,array2_size,((mean1-mean2)/sqrt(variance1/array1_size+variance2/array2_size)),return_value);
				}
				*/
				
			}
		}
		
	}	
}
