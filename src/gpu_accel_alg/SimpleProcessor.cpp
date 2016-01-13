#include "SimpleProcessor.h"
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cstdlib>


void SimpleProcessor::setNumberOfCores(int numberOfCores){
    this->numberOfCores = numberOfCores;
}

int SimpleProcessor::getNumberOfFeatureSizeTimesSampleSize2dArrays(int numOfFeatures){
	return numOfFeatures;
}


int SimpleProcessor::getNumberOfCores(){
	int numCPU = this->numberOfCores;
	
	if (numCPU > 0){
		return numCPU;
	}else{
		try{
			numCPU = sysconf(_SC_NPROCESSORS_ONLN);
		}catch( const std::exception& e ) { // reference to the base of a polymorphic object
			std::cout << e.what() <<std::endl; // information from length_error printed		
			numCPU = 1;
		}
		return numCPU;
	}
}
