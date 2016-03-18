#include "Processor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

using namespace std;

void Processor::setDebug(bool debug){
	this->debug = debug;
}
bool Processor::isDebugEnabled(){
	return this->debug;
}

int Processor::getFeaturesPerArray(int numOfFeatures, int arrayNumbers){	
	return ceil((numOfFeatures/(float)arrayNumbers));	
}

int Processor::getParallelizationType(){
	return parallelizationType;
}

void setBitOfInt(int& i, int pos, bool bit) 
{
	if(bit)
	{
		i |= (1 << pos);
	}
	else
	{
		i &= ~(1 << pos);
	}
}

int* Processor::packSampleFeatureMatrix(int numOfSamples, int numOfFeatures, char* sampleFeatureMatrix, bool* featureMask)
{
	//cout <<"numOfFeatures / 16 + (numOfFeatures % 16 == 0? 0 : 1)="<<(numOfFeatures / 16 + (numOfFeatures % 16 == 0? 0 : 1))<<endl;		
	//int intsPerInstance = numOfFeatures / 16 + (numOfFeatures % 16 == 0? 0 : 1);
	int intsPerInstance = (int)ceil((float)numOfFeatures / 16);		
	//int samplePerThread = (int)ceil(((float)numOfSamples)/threadSize);
	if(isDebugEnabled()){
		cout<<"intsPerInstance * numOfSamples="<<(intsPerInstance * numOfSamples)<<endl;
	}
	int* packeds = new int[intsPerInstance * numOfSamples];
	
	for (int s=0; s<numOfSamples; s++){
	
		int packedIndex = 0;			
		for(int i = 0; i < numOfFeatures; i += 16){		
			int packed = 0;
			for(int j = 0; j < 16; j++)
			{
				if(i + j < numOfFeatures)
				{
					int bit = sampleFeatureMatrix[s * numOfFeatures + i + j];
					if(featureMask[i+j] != true){
						bit = 0;
					}
					switch(bit)
					{
					case 0:
						break;
					case 1:
						setBitOfInt(packed, j * 2, 1);
						break;
					case 2:
						setBitOfInt(packed, j * 2 + 1, 1);
						break;
					case 3:
						setBitOfInt(packed, j * 2 + 1, 1);
						setBitOfInt(packed, j * 2, 1);
						break;
					default:
						std::cerr << "Something's wrong on the data" << std::endl;
						break;
					}
				}
			}
			
			packeds[s*intsPerInstance + packedIndex] = packed;
			packedIndex += 1;			
		}
	}		
	
	return packeds;
}
