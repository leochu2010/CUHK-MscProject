#include "SimpleTTestProcessor.h"
#include "utils/Timer.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <pthread.h>
#include <cstdlib>

using namespace std;

double calculate_ttest(char *array1, int array1_size, char *array2, int array2_size) {	
			
	if (array1_size <= 1) {
		return 1.0;		
	}
	if (array2_size <= 1) {
		return 1.0;
	}
	double mean1 = 0.0;
	double mean2 = 0.0;	
	
	for (size_t x = 0; x < array1_size; x++) {
		mean1 += array1[x];	
	}

	for (size_t x = 0; x < array2_size; x++) {
		mean2 += array2[x];
	}
		
	if (mean1 == mean2) {
		return 1.0;		
	}

	mean1 /= array1_size;
	mean2 /= array2_size;
	
	double variance1 = 0.0, variance2 = 0.0;
	
	for (size_t x = 0; x < array1_size; x++) {
		variance1 += (mean1-array1[x])*(mean1-array1[x]);
	}
	for (size_t x = 0; x < array2_size; x++) {
		variance2 += (mean2-array2[x])*(mean2-array2[x]);
	}
	
	if ((variance1 == 0.0) && (variance2 == 0.0)) {
		return 1.0;
	}
	variance1 = variance1/(array1_size-1);
	variance2 = variance2/(array2_size-1);	
	const double WELCH_T_STATISTIC = (mean1-mean2)/sqrt(variance1/array1_size+variance2/array2_size);
	
	return WELCH_T_STATISTIC;
}

void SimpleTTestProcessor::calculateAFeature(
	char* label0SamplesArray, int numOfLabel0Samples,
	char* label1SamplesArray, int numOfLabel1Samples,
	double* score
	){	
		*score = calculate_ttest(label1SamplesArray, numOfLabel1Samples, label0SamplesArray, numOfLabel0Samples);		
}
