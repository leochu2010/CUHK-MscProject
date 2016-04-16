#include "SimpleReliefFProcessor.h"
#include "utils/Timer.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include "threadpool/ThreadPool.h"

using namespace std;

SimpleReliefFProcessor::SimpleReliefFProcessor(int kNearest){
	parallelizationType = PARALLELIZE_ON_STAGES;	
	kNearestInstance = kNearest;
}

int SimpleReliefFProcessor::getKNearest(){
	return kNearestInstance;
}

void generateDisatanceMatrix(		
		int numOfSamples,
		int intsPerInstance,
		int* packedSampleFeatureMatrix,
		int* distanceMatrix		
	){		
		
	for(int sample1Id = 0; sample1Id<numOfSamples; sample1Id++){
	
		for(int sample2Id = sample1Id; sample2Id<numOfSamples; sample2Id++){
						
			int distance = 0;
			for(int k = 0; k < intsPerInstance; k++){
				int first = packedSampleFeatureMatrix[sample1Id * intsPerInstance + k];
				int second = packedSampleFeatureMatrix[sample2Id * intsPerInstance + k];
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
			
			distanceMatrix[numOfSamples * sample1Id + sample2Id] = distance;
			//distanceMatrix[numOfSamples * sample2Id + sample1Id] = distance;
			//printf("sample1Id=%d, sample2Id=%d, distance=%d, distanceMatrix[numOfSamples * sample1Id + sample2Id]=%d\n", sample1Id, sample2Id, distance, distanceMatrix[numOfSamples * sample1Id + sample2Id]);
			
		}
	}
}

int iParent(int id){
	return (id-1) / 2;
}

int iLeftChild(int id){
	return 2 * id + 1;
}

int iRightChild(int id){
	return 2 * id + 2;
}

void swap_items(int* array, int id1, int id2, int count, int array_offset){
	int offset = count * array_offset;
	int old_id = array[offset + id1];
	array[offset + id1]= array[offset + id2];
	array[offset + id2]= old_id;
}

int array_value(int *array, int *value, int id, int count, int array_offset){	
	//numOfSamples = count+1
	//sample1Id = array_offset
	//sample2Id = array[sample1Id * (numOfSamples-1) + id]
	/*
	if(array_offset == 0){	
		printf("array_offset=%d, count + 1=%d, id=%d, array[array_offset * count + id] = %d, valued=%d \n",array_offset,count+1,id,array[array_offset * count + id],value[array_offset * (count+1) + array[array_offset * count + id]]);			
	}
	*/	
	int sample1Id = array_offset;
	int sample2Id = array[array_offset * count + id];
	if(sample1Id < sample2Id){
		return value[sample1Id * (count+1) + sample2Id];
	}else{
		return value[sample2Id * (count+1) + sample1Id];
	}	
}

void shiftDown(int *array, int start, int end, int *value, int count, int array_offset){
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
	
	while (iLeftChild(root) <= end){
		child = iLeftChild(root);
		swap = root;
		
		if(array_value(array, value, swap, count, array_offset) < array_value(array, value, child, count, array_offset)){
			swap = child;
		}
		
		if((child + 1 <= end) && (array_value(array, value, swap, count, array_offset) < array_value(array, value, child + 1, count, array_offset))){
			swap = child + 1;
		}
		
		if(swap == root){
			return;
		}else{
			/*
			if(array_offset == 0){
				printf("before swap: array[root]=%d, array[swap]=%d\n",array[array_offset * count + root],array[array_offset * count + swap]);
			}*/
			swap_items(array, root, swap, count, array_offset);
			/*
			if(array_offset == 0){
				printf("after swap: array[root]=%d, array[swap]=%d\n",array[array_offset * count + root],array[array_offset * count + swap]);
			}
			*/
			root = swap;
		}
	}
}

void heapify(int* array, int count, int* value, int array_offset){
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
	int start = iParent(count - 1);
	
	while (start >= 0){
		shiftDown(array, start, count - 1, value, count, array_offset);
		start = start - 1;
	}
}

void heapSort(int *array, int count, int *value, int array_offset){

    //(Build the heap in array a so that largest value is at the root)
    heapify(array, count, value, array_offset);

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
		swap_items(array, end, 0, count, array_offset);	
		end = end - 1;
		shiftDown(array, 0, end, value, count, array_offset);
	}	
	
}

void heapSortDistance(
		int numOfSamples,				
		int* distanceHeaps,		
		int* distanceMatrix		
	){
		
	for(int sample1Id=0;sample1Id<numOfSamples;sample1Id++){
		for(int sample2Id=0;sample2Id<numOfSamples;sample2Id++){
			//put sample Id into heap
			//prepare heap sort here
			if(sample2Id > sample1Id){
				distanceHeaps[(numOfSamples-1) * sample1Id + sample2Id-1] = sample2Id;				
			}else if(sample2Id < sample1Id){
				distanceHeaps[(numOfSamples-1) * sample1Id + sample2Id] = sample2Id;			
			}
		}
	}

	for(int sample1Id=0; sample1Id < numOfSamples; sample1Id++){
		
		/*
		if(sample1Id == 999){
			for(int i=0;i<numOfSamples-1;i++){				
				int heapSampleId = distanceHeaps[(numOfSamples-1)*sample1Id+i];
				printf("before sorting: i=%d, distance=%d\n", i, distanceMatrix[numOfSamples*sample1Id+heapSampleId]);
			}
		}
		*/	
		
		heapSort(distanceHeaps, numOfSamples-1, distanceMatrix, sample1Id);
			
		/*
		if(sample1Id == 999){
			for(int i=0;i<numOfSamples-1;i++){
				int heapSampleId = distanceHeaps[(numOfSamples-1)*sample1Id+i];
				printf("after sorting: i=%d, sampleId=%d, distance=%d\n", i, heapSampleId, distanceMatrix[numOfSamples*sample1Id+heapSampleId]);
			}
		}
		*/
	}
		
}

void findKNearest(int numOfSamples, int sampleId, int* distanceHeaps, int* kNearestHit, int* kNearestMiss, char* labels, int kNearest){
	int numOfHitNearSamples = 0;
	int numOfMissNearSamples = 0;
	
	//heap length = numOfSamples - 1
	for(int i=(numOfSamples -1) * sampleId; i<(numOfSamples -1) * (sampleId+1); i++){
		int nearSampleId = distanceHeaps[i];
		
		if(numOfHitNearSamples == kNearest && numOfMissNearSamples == kNearest){
			return;
		}
		
		if(labels[sampleId] == labels[nearSampleId]){
			
			if(numOfHitNearSamples == kNearest){
				continue;
			}
			
			/*
			if(sampleId == 0){
				printf("hit near sampleId=%d\n",nearSampleId);
			}
			*/
			kNearestHit[sampleId * kNearest + numOfHitNearSamples] = nearSampleId;
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
			
			kNearestMiss[sampleId * kNearest + numOfMissNearSamples] = nearSampleId;
			numOfMissNearSamples += 1;
		}
	}
	
}

void weightFeatures(
		int kNearest,
		int numOfFeatures,
		int numOfSamples,
		int intsPerInstance,
		char* labels,
		int* kNearestHit,
		int* kNearestMiss,
		int* distanceHeaps,		
		bool* featureMask,
		int* packedSampleFeatureMatrix,
		float* weight,
		float* finalWeight
	){
	
	for(int sampleId=0; sampleId < numOfSamples; sampleId++){	
	
		findKNearest(numOfSamples, sampleId, distanceHeaps, kNearestHit, kNearestMiss, labels, kNearest);	
		
		/*
		if(sampleId == 999){
			for(int i=0;i<kNearest;i++){
				printf("k=%d, hitSampleId=%d, missSampleId=%d\n", i, kNearestHit[sampleId * kNearest+i], kNearestMiss[sampleId * kNearest+i]);
			}
		}
		*/
			
		for(int k=0; k<kNearest; k++){
			int hitSampleId = kNearestHit[sampleId * kNearest + k];
			int missSampleId = kNearestMiss[sampleId * kNearest + k];
			
			//cout<<"sampleId="<<sampleId<<", k="<<k<<", hit="<<hitSampleId<<", miss="<<missSampleId<<endl;
			
			for(int i=0;i<intsPerInstance;i++){
				int instanceInt = packedSampleFeatureMatrix[sampleId * intsPerInstance + i];
				int hitInt = packedSampleFeatureMatrix[hitSampleId * intsPerInstance + i];
				int missInt = packedSampleFeatureMatrix[missSampleId * intsPerInstance + i];

				for(int offset = 0; offset < 16; offset++)
				{
					int attributeIdx = i * 16 + offset;
					if(featureMask[attributeIdx] != true){
						continue;
					}
					
					if(attributeIdx < numOfFeatures)
					{
						int deltaHit = ((instanceInt >> offset * 2) & 0x3) == ((hitInt >> offset * 2) & 0x3)? 0 : 1;
						int deltaMiss = ((instanceInt >> offset * 2) & 0x3) == ((missInt >> offset * 2) & 0x3)? 0 : 1;
						float score = deltaMiss - deltaHit;
						//cout<<"sampleId="<<sampleId<<", k="<<k<<", hit="<<hitSampleId<<", miss="<<missSampleId<<", attributeIdx="<<attributeIdx <<", delteHit="<<deltaHit<<", deltaMiss="<<deltaMiss<<", score="<<score<<" ,weight="<<weight[sampleId * numOfFeatures + attributeIdx];
						weight[sampleId * numOfFeatures + attributeIdx] += score;					
						//cout<<"-->"<<weight[sampleId * numOfFeatures + attributeIdx]<<endl;;						
					}					
				}
			}			
		}
		
		for(int i=0; i<numOfFeatures; i++){		
			finalWeight[i] += weight[sampleId * numOfFeatures + i];
		}
	}
	
}

void SimpleReliefFProcessor::calculateAllFeatures(
	int numOfSamples, int numOfFeatures, 
	char* sampleFeatureMatrix, int* packedSampleFeatureMatrix,
	bool* featureMask, char* labels,
	double* scores, bool* success, string* errorMessage){
	
	int kNearest = getKNearest();	
	int intsPerInstance = (int)ceil((float)numOfFeatures / 16);	
	
	int* distanceHeaps = (int*)calloc(numOfSamples * (numOfSamples-1), sizeof(int));
	int* distanceMatrix = (int*)calloc(numOfSamples * numOfSamples, sizeof(int));
			
	int* kNearestHit = (int*)calloc(kNearest * numOfSamples, sizeof(int));
	int* kNearestMiss = (int*)calloc(kNearest * numOfSamples, sizeof(int));	
	
	float* finalWeight = (float*)calloc(numOfFeatures,sizeof(float));
	float* weight = (float*)calloc(numOfSamples*numOfFeatures,sizeof(float));	
	
	if(isDebugEnabled()){
		cout<<"generate distance matrix"<<endl;
	}
	
	generateDisatanceMatrix(		
		numOfSamples,
		intsPerInstance,
		packedSampleFeatureMatrix,
		distanceMatrix
	);
	
	if(isDebugEnabled()){
		cout<<"heap sort distance"<<endl;
	}
	
	heapSortDistance(
		numOfSamples,		
		distanceHeaps,
		distanceMatrix		
	);
	
	if(isDebugEnabled()){
		cout<<"weight features"<<endl;
	}	
		
	weightFeatures(
		kNearest,
		numOfFeatures,
		numOfSamples,
		intsPerInstance,
		labels,
		kNearestHit,
		kNearestMiss,
		distanceHeaps,		
		featureMask,
		packedSampleFeatureMatrix,
		weight,
		finalWeight		
	);
	
	if(isDebugEnabled()){
		cout<<"generate result"<<endl;
	}
	
	int divisor = numOfSamples * kNearest;
	for(int i=0;i<numOfFeatures;i++){
		scores[i] = finalWeight[i]/divisor;
	}
	
	if(isDebugEnabled()){
		cout<<"free memory"<<endl;
	}
	
	free(distanceHeaps);
	free(distanceMatrix);
	free(kNearestHit);
	free(kNearestMiss);
	free(weight);
	free(finalWeight);
		
	*success = true;
}