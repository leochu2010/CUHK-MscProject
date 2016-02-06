#include "SimpleReliefFProcessor.h"
#include "utils/Timer.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <pthread.h>
#include <cstdlib>

using namespace std;

SimpleReliefFProcessor::SimpleReliefFProcessor(){
	parallelizationType = PARALLELIZE_ON_STAGES;	
}


void SimpleReliefFProcessor::calculateAllFeatures(
			char** label0SamplesArray, int numOfLabel0Samples,
			char** label1SamplesArray, int numOfLabel1Samples,
			int numOfFeatures,
			double* scores, bool* success, string* errorMessage){
	
	
	/*
		all dists 1d array?
		yes
		thread pool		
		iteration weight
	*/
}