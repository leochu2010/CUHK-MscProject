#include "CpuProcessor.h"
#include <stdlib.h>
#include <iostream>
#include <unistd.h>


void CpuProcessor::setNumberOfThreads(int numberOfThreads){
    this->numberOfThreads = numberOfThreads;
}

int CpuProcessor::getNumberOfFeatureSizeTimesSampleSize2dArrays(int numOfFeatures){
	return numOfFeatures;
}

/*
int CpuProcessor::getNumberOfProcessingUnit(){
	int numCPU = 1;
	
	try{
		numCPU = sysconf(_SC_NPROCESSORS_ONLN);
	}catch( const std::exception& e ) { // reference to the base of a polymorphic object
		std::cout << e.what() <<std::endl; // information from length_error printed		
		numCPU = 1;
	}
	
	return numCPU;
}
*/