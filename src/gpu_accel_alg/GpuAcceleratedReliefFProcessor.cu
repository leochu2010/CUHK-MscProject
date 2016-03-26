#include "GpuAcceleratedReliefFProcessor.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "utils/Timer.h"

using namespace std;


GpuAcceleratedReliefFProcessor::GpuAcceleratedReliefFProcessor(int kNearest){
	parallelizationType = PARALLELIZE_ON_STAGES;
	kNearestInstance = kNearest;
}

int GpuAcceleratedReliefFProcessor::getKNearest(){
	return kNearestInstance;
}

__global__ void gpu_generateDisatanceMatrix(
		int samplePerThread,
		int numOfSamples,			
		int numOfFeatures,		
		int intsPerInstance,		
		int* d_packedSampleFeatureMatrix,		
		int* d_distanceMatrix,
		int* d_distanceHeaps
	){
		
	int sample1Id = gridDim.x * blockIdx.x + blockIdx.y;	

	if(sample1Id >= numOfSamples){
		return;
	}

	/*
	if(threadIdx.x == 0){
		printf("sample1=%d, gridDim.x=%d * blockIdx.x=%d + blockIdx.y=%d\n", sample1Id,gridDim.x,blockIdx.x,blockIdx.y);
	}
	*/
	
	//put distance into an array	
	
	for(int i = 0; i < samplePerThread; i++){
		
		int sample2Id = threadIdx.x * samplePerThread + i;

		if(sample2Id >= numOfSamples){
			break;
		}
		
		//put sample Id into heap
		//prepare heap sort here
		if(sample2Id > sample1Id){
			d_distanceHeaps[(numOfSamples-1) * sample1Id + sample2Id-1] = sample2Id;				
		}else if(sample2Id < sample1Id){
			d_distanceHeaps[(numOfSamples-1) * sample1Id + sample2Id] = sample2Id;			
		}
		
		if(sample2Id < sample1Id){
			continue;
		}
		
		int distance = 0;
		for(int k = 0; k < intsPerInstance; k++){
			int first = d_packedSampleFeatureMatrix[sample1Id * intsPerInstance + k];
			int second = d_packedSampleFeatureMatrix[sample2Id * intsPerInstance + k];
			/*
			if(threadIdx.x == 0 && k==0){
				printf("first=%d, second=%d\n", first, second);
			}*/
			int ret = first ^ second;
			for(int l = 0; l < 32; l += 2){
				int diff = (ret >> l) & 3;
				if(diff != 0) distance++;
			}
		}
		
		d_distanceMatrix[numOfSamples * sample1Id + sample2Id] = distance;
		d_distanceMatrix[numOfSamples * sample2Id + sample1Id] = distance;

	}
}

__device__ int gpu_iParent(int id){
	return (id-1) / 2;
}

__device__ int gpu_iLeftChild(int id){
	return 2 * id + 1;
}

__device__ int gpu_iRightChild(int id){
	return 2 * id + 2;
}

__device__ void gpu_swap(int* d_array, int id1, int id2, int count, int array_offset){
	int offset = count * array_offset;
	int old_id = d_array[offset + id1];
	d_array[offset + id1]= d_array[offset + id2];
	d_array[offset + id2]= old_id;
}

__device__ int gpu_array_value(int *d_array, int *d_value, int id, int count, int array_offset){	
	//numOfSamples = count+1
	//sample1Id = array_offset
	//sample2Id = d_array[sample1Id * (numOfSamples-1) + id]
	/*
	if(array_offset == 0){	
		printf("array_offset=%d, count + 1=%d, id=%d, d_array[array_offset * count + id] = %d, d_valued=%d \n",array_offset,count+1,id,d_array[array_offset * count + id],d_value[array_offset * (count+1) + d_array[array_offset * count + id]]);			
	}
	*/	
	return d_value[array_offset * (count+1) + d_array[array_offset * count + id]];		
}

__device__ void gpu_shiftDown(int *d_array, int start, int end, int *d_value, int count, int array_offset){
	/*root ← start

    while iLeftChild(root) ≤ end do    (While the root has at least one child)
        child ← iLeftChild(root)   (Left child of root)
        swap ← root                (Keeps track of child to swap with)

        if a[swap] < a[child]
            swap ← child
        (If there is a right child and that child is greater)
        if child+1 ≤ end and a[swap] < a[child+1]
            swap ← child + 1
        if swap = root
            (The root holds the largest element. Since we assume the heaps rooted at the
             children are valid, this means that we are done.)
            return
        else
            swap(a[root], a[swap])
            root ← swap            (repeat to continue sifting down the child now)
	*/
	
	int root = start;
	int swap;
	int child;
	
	while (gpu_iLeftChild(root) <= end){
		child = gpu_iLeftChild(root);
		swap = root;
		
		if(gpu_array_value(d_array, d_value, swap, count, array_offset) < gpu_array_value(d_array, d_value, child, count, array_offset)){
			swap = child;
		}
		
		if((child + 1 <= end) && (gpu_array_value(d_array, d_value, swap, count, array_offset) < gpu_array_value(d_array, d_value, child + 1, count, array_offset))){
			swap = child + 1;
		}
		
		if(swap == root){
			return;
		}else{
			/*
			if(array_offset == 0){
				printf("before swap: d_array[root]=%d, d_array[swap]=%d\n",d_array[array_offset * count + root],d_array[array_offset * count + swap]);
			}*/
			gpu_swap(d_array, root, swap, count, array_offset);
			/*
			if(array_offset == 0){
				printf("after swap: d_array[root]=%d, d_array[swap]=%d\n",d_array[array_offset * count + root],d_array[array_offset * count + swap]);
			}
			*/
			root = swap;
		}
	}
}

__device__ void gpu_heapify(int* d_array, int count, int* d_value, int array_offset){
	/*
	 (start is assigned the index in 'a' of the last parent node)
    (the last element in a 0-based array is at index count-1; find the parent of that element)
    start ← iParent(count-1)
    
    while start ≥ 0 do
        (sift down the node at index 'start' to the proper place such that all nodes below
         the start index are in heap order)
        siftDown(a, start, count - 1)
        (go to the next parent node)
        start ← start - 1
    (after sifting down the root all nodes/elements are in heap order)
	*/
	int start = gpu_iParent(count - 1);
	
	while (start >= 0){
		gpu_shiftDown(d_array, start, count - 1, d_value, count, array_offset);
		start = start - 1;
	}
}

__device__ void gpu_heapSort(int *d_array, int count, int *d_value, int array_offset){

    //(Build the heap in array a so that largest value is at the root)
    gpu_heapify(d_array, count, d_value, array_offset);

    /*(The following loop maintains the invariants that a[0:end] is a heap and every element
     beyond end is greater than everything before it (so a[end:count] is in sorted order))
	 
	 end ← count - 1
	
    while end > 0 do
        (a[0] is the root and largest value. The swap moves it in front of the sorted elements.)
        swap(a[end], a[0])
        (the heap size is reduced by one)
        end ← end - 1
        (the swap ruined the heap property, so restore it)
        siftDown(a, 0, end)
	*/
	
	
	int end = count - 1;
	while (end >0){		
		gpu_swap(d_array, end, 0, count, array_offset);	
		end = end - 1;
		gpu_shiftDown(d_array, 0, end, d_value, count, array_offset);
	}	
	
}

__global__ void gpu_heapSortDistance(
		int numOfSamples,		
		char* d_labels,
		int* d_distanceHeaps,		
		int* d_distanceMatrix
	){

	int sample1Id = gridDim.x * blockIdx.x + blockIdx.y;	

	if(sample1Id >= numOfSamples){
		return;
	}
	
	/*
	if(sample1Id == 999){
		for(int i=0;i<numOfSamples-1;i++){				
			int heapSampleId = d_distanceHeaps[(numOfSamples-1)*sample1Id+i];
			printf("before sorting: i=%d, distance=%d\n", i, d_distanceMatrix[numOfSamples*sample1Id+heapSampleId]);
		}
	}
	*/	
	
	gpu_heapSort(d_distanceHeaps, numOfSamples-1, d_distanceMatrix, sample1Id);
		
	/*
	if(sample1Id == 999){
		for(int i=0;i<numOfSamples-1;i++){
			int heapSampleId = d_distanceHeaps[(numOfSamples-1)*sample1Id+i];
			printf("after sorting: i=%d, sampleId=%d, distance=%d\n", i, heapSampleId, d_distanceMatrix[numOfSamples*sample1Id+heapSampleId]);
		}
	}
	*/
		
}

__device__ void gpu_findKNearest(int numOfSamples, int sampleId, int* d_distanceHeaps, int* d_kNearestHit, int* d_kNearestMiss, char* d_labels, int kNearest){
	int numOfHitNearSamples = 0;
	int numOfMissNearSamples = 0;
	
	//heap length = numOfSamples - 1
	for(int i=(numOfSamples -1) * sampleId; i<(numOfSamples -1) * (sampleId+1); i++){
		int nearSampleId = d_distanceHeaps[i];
		
		if(numOfHitNearSamples == kNearest && numOfMissNearSamples == kNearest){
			return;
		}
		
		if(d_labels[sampleId] == d_labels[nearSampleId]){
			
			if(numOfHitNearSamples == kNearest){
				continue;
			}
			
			/*
			if(sampleId == 0){
				printf("hit near sampleId=%d\n",nearSampleId);
			}
			*/
			d_kNearestHit[sampleId * kNearest + numOfHitNearSamples] = nearSampleId;
			numOfHitNearSamples += 1;
		}else{
			if(numOfMissNearSamples == kNearest){
				continue;
			}
			
			/*
			if(sampleId == 0){
				printf("miss near sampleId=%d\n",nearSampleId);
			}
			*/
			
			d_kNearestMiss[sampleId * kNearest + numOfMissNearSamples] = nearSampleId;
			numOfMissNearSamples += 1;
		}
	}
	
}

__global__ void gpu_weightFeatures(
		int kNearest,
		int numOfFeatures,
		int numOfSamples,
		int intsPerInstance,
		char* d_labels,
		int* d_kNearestHit,
		int* d_kNearestMiss,
		int* d_distanceHeaps,		
		bool* d_featureMask,
		int* d_packedSampleFeatureMatrix,
		float* d_weight,
		float* d_finalWeight
	){
		
	int sampleId = gridDim.x * blockIdx.x + blockIdx.y;
	
	//printf("sample=%d, gridDim.x=%d * blockIdx.x=%d + blockIdx.y=%d\n", sampleId,gridDim.x,blockIdx.x,blockIdx.y);	
	
	
	if(sampleId >= numOfSamples){
		return;
	}
	
	gpu_findKNearest(numOfSamples, sampleId, d_distanceHeaps, d_kNearestHit, d_kNearestMiss, d_labels, kNearest);	
	
	/*
	if(sampleId == 999){
		for(int i=0;i<kNearest;i++){
			printf("k=%d, hitSampleId=%d, missSampleId=%d\n", i, d_kNearestHit[sampleId * kNearest+i], d_kNearestMiss[sampleId * kNearest+i]);
		}
	}
	*/
		
	for(int k=0; k<kNearest; k++){		
		int hitSampleId = d_kNearestHit[sampleId * kNearest + k];
		int missSampleId = d_kNearestMiss[sampleId * kNearest + k];
		
		for(int i=0;i<intsPerInstance;i++){
			int instanceInt = d_packedSampleFeatureMatrix[sampleId * intsPerInstance + i];
			int hitInt = d_packedSampleFeatureMatrix[hitSampleId * intsPerInstance + i];
			int missInt = d_packedSampleFeatureMatrix[missSampleId * intsPerInstance + i];

			for(int offset = 0; offset < 16; offset++)
			{
				int attributeIdx = i * 16 + offset;
				if(d_featureMask[attributeIdx] != true){
					continue;
				}
				
				if(attributeIdx < numOfFeatures)
				{
					int deltaHit = ((instanceInt >> offset * 2) & 0x3) == ((hitInt >> offset * 2) & 0x3)? 0 : 1;
					int deltaMiss = ((instanceInt >> offset * 2) & 0x3) == ((missInt >> offset * 2) & 0x3)? 0 : 1;
					float score = deltaMiss - deltaHit;
					d_weight[sampleId * numOfFeatures + attributeIdx] += score;					
				}
			}
		}
	}
	
	for(int i=0; i<numOfFeatures; i++){		
		atomicAdd(&d_finalWeight[i],d_weight[sampleId * numOfFeatures + i]);
	}
	
}

Result* GpuAcceleratedReliefFProcessor::parallelizeCalculationOnStages(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, int* packedSampleFeatureMatrix, bool* featureMask, char* labels){
	
	if(isDebugEnabled()){
		cout<<"numOfSamples="<<numOfSamples<<", numOfFeatures="<<numOfFeatures<<endl;
	}
	
	Timer processing("Processing");
	processing.start();
	
	int kNearest = getKNearest();
	
	bool* d_featureMask;
	int* d_packedSampleFeatureMatrix;
	int* d_distanceHeaps;	
	int* d_distanceMatrix;	
	char* d_labels;
		
	//int intsPerInstance = numOfFeatures / 16 + (numOfFeatures % 16 == 0? 0 : 1);
	int intsPerInstance = (int)ceil((float)numOfFeatures / 16);	
	
	cudaMalloc(&d_featureMask, numOfFeatures*sizeof(bool));
	cudaMemcpy(d_featureMask, featureMask, numOfFeatures*sizeof(bool),cudaMemcpyHostToDevice);
	getMemoryInfo("after featureMask cudaMalloc");
	
	cudaMalloc(&d_packedSampleFeatureMatrix, intsPerInstance * numOfSamples*sizeof(int));	
	cudaMemcpy(d_packedSampleFeatureMatrix, packedSampleFeatureMatrix, intsPerInstance * numOfSamples*sizeof(int),cudaMemcpyHostToDevice);	
	getMemoryInfo("after packedSampleFeatureMatrix cudaMalloc");
	
	cudaMalloc(&d_labels, numOfSamples*sizeof(char));
	cudaMemcpy(d_labels, labels, numOfSamples*sizeof(char),cudaMemcpyHostToDevice);
	getMemoryInfo("after labels cudaMalloc");	
		
	cudaMalloc(&d_distanceHeaps, numOfSamples * (numOfSamples-1) * sizeof(int));	
	getMemoryInfo("after distanceHeaps cudaMalloc");
	
	cudaMalloc(&d_distanceMatrix, numOfSamples * numOfSamples * sizeof(int));
	getMemoryInfo("after distanceMatrix cudaMalloc");
	
	int* d_kNearestHit;
	int* d_kNearestMiss;
	cudaMalloc(&d_kNearestHit, kNearest * numOfSamples*sizeof(int));
	getMemoryInfo("after kNearestHit cudaMalloc");
	cudaMalloc(&d_kNearestMiss, kNearest * numOfSamples*sizeof(int));
	getMemoryInfo("after kNearestMiss cudaMalloc");
		
	float* finalWeight = (float*)calloc(numOfFeatures,sizeof(float));
	float* d_weight;
	float* d_finalWeight;
	
	cudaMalloc(&d_weight, numOfSamples*numOfFeatures*sizeof(float));
	cudaMemset(d_weight, 0, numOfSamples*numOfFeatures*sizeof(float));
	getMemoryInfo("after weight cudaMalloc");
	
	
	cudaMalloc(&d_finalWeight, numOfFeatures*sizeof(float));	
	cudaMemset(d_finalWeight, 0, numOfFeatures*sizeof(float));
	getMemoryInfo("after finalWeight cudaMalloc");
		
	int grid2d = (int)ceil(pow(numOfSamples,1/2.));
	int threadSize = getNumberOfThreadsPerBlock();
		
	int samplePerThread = (int)ceil(((float)numOfSamples)/threadSize);	
	
	if(isDebugEnabled()){
		cout<<"grid size="<<grid2d<<"x"<<grid2d<<endl;
		cout<<"thread size="<<threadSize<<endl;
		cout<<"samplePerThread="<<samplePerThread<<endl;
	}
	
	dim3 gridSize(grid2d,grid2d);
	
	if(isDebugEnabled()){
		cout<<"generate distance heaps"<<endl;
	}
	gpu_generateDisatanceMatrix<<<gridSize, threadSize>>>(		
		samplePerThread,
		numOfSamples,			
		numOfFeatures,		
		intsPerInstance,		
		d_packedSampleFeatureMatrix,
		d_distanceMatrix,
		d_distanceHeaps		
		);
	cudaDeviceSynchronize();		
	
	if(this->isDebugEnabled()){		
		cout<<"cudaPeekAtLastError:"<<cudaGetErrorString(cudaPeekAtLastError())<<endl;
	}
	
	if(isDebugEnabled()){
		cout<<"heap sort distance"<<endl;
	}	
	
	gpu_heapSortDistance<<<gridSize, 1>>>(
		numOfSamples,	
		d_labels,
		d_distanceHeaps,		
		d_distanceMatrix		
	);
	
	cudaDeviceSynchronize();		
	
	if(this->isDebugEnabled()){		
		cout<<"cudaPeekAtLastError:"<<cudaGetErrorString(cudaPeekAtLastError())<<endl;
	}
	
	if(isDebugEnabled()){
		cout<<"weight features"<<endl;
	}	
	
	gpu_weightFeatures<<<gridSize,1>>>(
		kNearest,
		numOfFeatures,
		numOfSamples,
		intsPerInstance,
		d_labels,
		d_kNearestHit,
		d_kNearestMiss,
		d_distanceHeaps,		
		d_featureMask,
		d_packedSampleFeatureMatrix,
		d_weight,
		d_finalWeight
	);
			
	cudaDeviceSynchronize();
	
	if(this->isDebugEnabled()){		
		cout<<"cudaPeekAtLastError:"<<cudaGetErrorString(cudaPeekAtLastError())<<endl;
	}
	
	cudaMemcpy(finalWeight, d_finalWeight, numOfFeatures*sizeof(float), cudaMemcpyDeviceToHost);
			
	if(isDebugEnabled()){
		cout<<"generate result"<<endl;
	}
	Result* result = new Result;
	result->scores = new double[numOfFeatures];
	int divider = numOfSamples * kNearest;
	for(int i=0;i<numOfFeatures;i++){
		result->scores[i] = finalWeight[i]/divider;
	}
	result->success = true;	
		
	free(finalWeight);
	
	cudaFree(d_packedSampleFeatureMatrix);	
	cudaFree(d_labels);
	cudaFree(d_distanceHeaps);	
	cudaFree(d_featureMask);	
	cudaFree(d_labels);
	cudaFree(d_kNearestHit);
	cudaFree(d_kNearestMiss);
	cudaFree(d_weight);
	cudaFree(d_finalWeight);
	
	processing.stop();
	result->startTime=processing.getStartTime();
	result->endTime=processing.getStopTime();	
	
	return result;
}

