#include "PValueGpuProcessor.h"
#include "Timer.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__global__ void calculate_Pvalue(
	char *d_array1, int array1_size, int array1_size_per_thread, size_t pitch0,
	char *d_array2, int array2_size, int array2_size_per_thread, size_t pitch1, 
	double *d_score, 
	int dev, int featuresPerDevice, 
	int numOfFeatures);

Result* PValueGpuProcessor::calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels)
{
	Timer t1 ("Total");
	t1.start();
	
	Timer pre("Pre-processing");
	pre.start();
	
	Result* testResult = new Result;
	testResult->scores=new double[numOfFeatures];
	
	if(this->isDebugEnabled()){
		std::cout<<std::endl;	
	}
	
	int label0Size = 0;
	int label1Size = 0;
	
	for(int j=0; j<numOfSamples; j++)
	{			
		if((int)labels[j]==0){
			label0Size+=1;		
		}else if((int)labels[j]==1){
			label1Size+=1;
		}
	}
	
	int deviceCount = getNumberOfDevice();
	
	int featuresPerDevice = round((numOfFeatures/(float)deviceCount)+0.5f);
	if(this->isDebugEnabled()){
		pre.printTimeSinceStart();
	}
	
	int threadSize = getNumberOfThreadsPerBlock();
	
	if(this->isDebugEnabled()){
		std::cout<<"deviceCount="<<deviceCount<<", threadSize="<<threadSize<<std::endl;	
	}
	
	int label0SizePerThread = round((label0Size/(float)(threadSize))+0.5f);
	int label1SizePerThread = round((label1Size/(float)(threadSize))+0.5f);
	
	char label0Array[deviceCount][featuresPerDevice][label0Size];
	char label1Array[deviceCount][featuresPerDevice][label1Size];
	double score[deviceCount][featuresPerDevice];
	
	for(int i=0;i<numOfFeatures;i++){
		int deviceId = i / featuresPerDevice;
		int featureId = i % featuresPerDevice;
		
		//std::cout<<"featuresPerDevice="<<featuresPerDevice<<", dev="<<deviceId<<", featureId="<<featureId<<std::endl;
		
		if(featureMask[i] != true){
			continue;
		}

		int label0Index=0;
		int label1Index=0;
		
		for(int j=0; j<numOfSamples; j++)
		{
			int index = j*numOfFeatures + i;			
			if(labels[j]==0){
				label0Array[deviceId][featureId][label0Index]=(char)sampleTimesFeature[index];
				label0Index+=1;
			}else if(labels[j]==1){
				label1Array[deviceId][featureId][label1Index]=(char)sampleTimesFeature[index];
				label1Index+=1;				
			}
		}
		
		score[deviceId][featureId]=0;
	}
	cudaProfilerStart();	
	Timer totalProcessing("Total Processing");
	totalProcessing.start();
	pre.stop();	
	
	size_t pitch0[deviceCount];
	size_t pitch1[deviceCount];
	char *d_label0Array[deviceCount];
	char *d_label1Array[deviceCount];
	double *d_score[deviceCount];
		
	Timer t3("Host to Device");
	t3.start();	
	for(int dev=0; dev<deviceCount; dev++) {
		cudaSetDevice(dev);
		
		cudaMallocPitch(&d_label0Array[dev],&pitch0[dev],label0Size*sizeof(char),featuresPerDevice); 
		cudaMallocPitch(&d_label1Array[dev],&pitch1[dev],label1Size*sizeof(char),featuresPerDevice); 
		
		cudaMemcpy2D(d_label0Array[dev],pitch0[dev],label0Array[dev],label0Size*sizeof(char),label0Size*sizeof(char),featuresPerDevice,cudaMemcpyHostToDevice);
		cudaMemcpy2D(d_label1Array[dev],pitch1[dev],label1Array[dev],label1Size*sizeof(char),label1Size*sizeof(char),featuresPerDevice,cudaMemcpyHostToDevice);
		
		cudaMalloc(&d_score[dev],featuresPerDevice*sizeof(double));
		
		cudaMemcpy(d_score[dev],score[dev],featuresPerDevice*sizeof(double),cudaMemcpyHostToDevice);
		//std::cout<<"device"<<dev<<" ";
		//t3.printTimeSinceStart();
	}	
	t3.stop();
		
	int grid2d = (int)round(pow(featuresPerDevice,1/2.)+0.5f);
	if(this->isDebugEnabled()){
		std::cout<<"featuresPerDevice="<<featuresPerDevice<<",grid2d="<<grid2d<<",label0SizePerThread="<<label0SizePerThread<<",label1SizePerThread="<<label1SizePerThread<<std::endl;
	}
			
	dim3 gridSize(grid2d,grid2d);
	   
	Timer t4 ("Processing Time");
	t4.start();
	for(int dev=0; dev<deviceCount; dev++) {
		cudaSetDevice(dev);
		calculate_Pvalue<<<gridSize, threadSize>>>(
			d_label1Array[dev], label1Size, label1SizePerThread, pitch1[dev], 
			d_label0Array[dev], label0Size, label0SizePerThread, pitch0[dev], 
			d_score[dev], 
			dev, featuresPerDevice, 
			numOfFeatures);
		//std::cout<<"device"<<dev<<" ";
		//t4.printTimeSinceStart();
	}
		
	if(this->isDebugEnabled()){
		std::cout<<"cudaPeekAtLastError:"<<cudaPeekAtLastError()<<std::endl;
	}
	cudaDeviceSynchronize();
	t4.stop();
		
	Timer t5 ("Device to Host");
	t5.start();
	for(int dev=0; dev<deviceCount; dev++) {
		cudaSetDevice(dev);
		cudaMemcpy(score[dev], d_score[dev], featuresPerDevice*sizeof(double), cudaMemcpyDeviceToHost);
	}
	t5.stop();
	
	cudaFree(d_label1Array);
	cudaFree(d_label0Array);
	cudaFree(d_score);
	
	cudaProfilerStop();
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
	t1.stop();
	
	if(this->isDebugEnabled()){
		pre.printTimeSpent();	
		t3.printTimeSpent();
		t4.printTimeSpent();
		t5.printTimeSpent();	
		totalProcessing.printTimeSpent();
		t1.printTimeSpent();
	}
	

	testResult->startTime=totalProcessing.getStartTime();
	testResult->endTime=totalProcessing.getStopTime();		
	
	return testResult;
}

__global__ void calculate_Pvalue(
	char *d_array1, int array1_size, int array1_size_per_thread, size_t pitch0,
	char *d_array2, int array2_size, int array2_size_per_thread, size_t pitch1, 
	double *d_score, 
	int dev, int featuresPerDevice, 
	int numOfFeatures) {
	//gridDim.x = gridDim.yas it is 2d 
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
			
	    char* array1 = (char*)((char*)d_array1 + idx*pitch0);
		char* array2 = (char*)((char*)d_array2 + idx*pitch1);
				
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
				m1+=array1[i];
				/*
				if(idx==0){
					printf("i=%d, value=%d\n",i,array1[i]);
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
				m2+=array2[i];
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
				v1=(mean1-array1[i]);
				v1s += v1*v1; 
			}			
			atomicAdd(&variance1, v1s);
		}			

		if(array2_size_per_thread*(threadIdx.x) < array2_size){
			float v2 = 0;
			float v2s = 0;
			for(int i=array2_size_per_thread*(threadIdx.x); i<array2_size_per_thread*(threadIdx.x+1) && i<array2_size; i++){
				v2=(mean2-array2[i]);
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
			const int N = 65535;
			h = x/N;
			sum1=0;
			sum2=0;
		}
		__syncthreads();
		
		
		
		const int N = 65535;
		int NPerThread = round(((float)N)/blockDim.x+0.5f);			
		float s1=0;
		float s2=0;
		for(unsigned int i = (threadIdx.x)*(NPerThread); i < (threadIdx.x+1)*(NPerThread); i++) {
			if(i<N){					
				s1 += (pow(h * i + h / 2.0,a-1))/(sqrt(1-(h * i + h / 2.0)));
				s2 += (pow(h * i,a-1))/(sqrt(1-h * i));
			}
		}
		atomicAdd(&sum1, s1);
		atomicAdd(&sum2, s2);		

		__syncthreads();
		
		/*		
		if(threadIdx.x == 0 && idx==0){
			printf("idx=%d: sum1=%f, sum2=%f, NPerThread=%d, threads=%d\n",idx,sum1,sum2,NPerThread, threads);			
		}
		*/
		
		
		if (threadIdx.x == 0){
			double return_value = ((h / 6.0) * ((pow(x,a-1))/(sqrt(1-x)) + 4.0 * sum1 + 2.0 * sum2))/(exp(lgamma(a)+0.57236494292470009-lgamma(a+0.5)));			
			if ((isfinite(return_value) == 0) || (return_value > 1.0)) {
				d_score[idx] = 1.0;		
			} else {							
				d_score[idx] = return_value;
				//if (dev==0){
					//printf("idx=%d: sum1=%f, sum2=%f\n",idx,sum1,sum2);
					//printf("idx=%d: score=%f\n",dev*featuresPerDevice+idx,return_value);
				//}
			}
		}
		
	}	
}
