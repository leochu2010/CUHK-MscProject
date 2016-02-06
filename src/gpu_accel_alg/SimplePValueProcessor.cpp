#include "SimplePValueProcessor.h"
#include "utils/Timer.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <pthread.h>
#include <cstdlib>

using namespace std;

SimplePValueProcessor::SimplePValueProcessor(){
	parallelizationType = PARALLELIZE_ON_FEATURES;	
}

double calculate_Pvalue(char *array1, int array1_size, char *array2, int array2_size) {	
			
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
		
	const double DEGREES_OF_FREEDOM = pow((variance1/array1_size+variance2/array2_size),2.0)//numerator
	 /
	(
		(variance1*variance1)/(array1_size*array1_size*(array1_size-1))+
		(variance2*variance2)/(array2_size*array2_size*(array2_size-1))
	);
	const double a = DEGREES_OF_FREEDOM/2, x = DEGREES_OF_FREEDOM/(WELCH_T_STATISTIC*WELCH_T_STATISTIC+DEGREES_OF_FREEDOM);
	const unsigned short int N = 65535;
	const double h = x/N;
	double sum1 = 0.0, sum2 = 0.0;
	
	for(unsigned short int i = 0;i < N; i++) {
      sum1 += (pow(h * i + h / 2.0,a-1))/(sqrt(1-(h * i + h / 2.0)));
      sum2 += (pow(h * i,a-1))/(sqrt(1-h * i));
	}
	
	double return_value = ((h / 6.0) * ((pow(x,a-1))/(sqrt(1-x)) + 4.0 * sum1 + 2.0 * sum2))/(expl(lgammal(a)+0.57236494292470009-lgammal(a+0.5)));
		
	if ((isfinite(return_value) == 0) || (return_value > 1.0)) {
		return 1.0;
	} else {
		return return_value;		
	}
}

void SimplePValueProcessor::calculateAFeature(
	char* label0SamplesArray, int numOfLabel0Samples,
	char* label1SamplesArray, int numOfLabel1Samples,
	double* score
	){	
		*score = calculate_Pvalue(label1SamplesArray, numOfLabel1Samples, label0SamplesArray, numOfLabel0Samples);		
}
