#include "PValueCpuProcessor.h"
#include "utils/Timer.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "lib/ThreadPool/ThreadPool.h"

int myTask(){
	return 1;
}

Result* PValueCpuProcessor::calculate(int numOfFeatures, 
	char** label0FeatureSizeTimesSampleSize2dArray, int numOfLabel0Samples,
	char** label1FeatureSizeTimesSampleSize2dArray, int numOfLabel1Samples, 
	bool* featureMask){
		
	
	    ThreadPool pool(4);
    std::vector< std::future<int> > results;

    for(int i = 0; i < 8; ++i) {
        results.emplace_back(
            pool.enqueue([i] {
                std::cout << "hello " << i << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                std::cout << "world " << i << std::endl;
                return i*i;
            })
        );
    }

    for(auto && result: results)
        std::cout << result.get() << ' ';
    std::cout << std::endl;
	
		
	Timer t1("Processing");
	t1.start();		
			
	Result* testResult = new Result;
	testResult->scores=new double[numOfFeatures];
	
	for(int i=0;i<numOfFeatures;i++){
		if(featureMask[i] != true){			
			continue;
		}		
		double score = this->calculate_Pvalue(label1FeatureSizeTimesSampleSize2dArray[i], numOfLabel1Samples, label0FeatureSizeTimesSampleSize2dArray[i], numOfLabel0Samples);
		testResult->scores[i]=score;
		std::cout<<"Feature "<<i<<":"<<score<<std::endl;		
	}
	t1.stop();
	
	testResult->success = true;
	testResult->startTime=t1.getStartTime();
	testResult->endTime=t1.getStopTime();
	return testResult;	
		
}
/*

Result* PValueCpuProcessor::calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels)
{
	Timer t1 ("Total");
	t1.start();
	
	
	Result* testResult = new Result;
	testResult->scores=new double[numOfFeatures];
	
	std::cout<<std::endl;	
	
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
	
	for(int i=0;i<numOfFeatures;i++)
	{
		if(featureMask[i] != true){
			continue;
		}
						
		double label0Array[label0Size];
		double label1Array[label1Size];
		int label0Index=0;
		int label1Index=0;
				
		for(int j=0; j<numOfSamples; j++)
		{
			int index = j*numOfFeatures + i;			
			if(labels[j]==0){
				label0Array[label0Index]=(int)sampleTimesFeature[index];
				label0Index+=1;
			}else if(labels[j]==1){
				label1Array[label1Index]=(int)sampleTimesFeature[index];
				label1Index+=1;
			}
		}
				
		double score = this->calculate_Pvalue(label1Array, label1Size, label0Array, label0Size);

		testResult->scores[i]=score;
		//std::cout<<"Feature "<<i<<":"<<score<<std::endl;		
	}
	
	std::cout<<std::endl;
		
	t1.stop();
	t1.printTimeSpent();
	

	testResult->startTime=t1.getStartTime();
	testResult->endTime=t1.getStopTime();
	return testResult;
}
*/

double PValueCpuProcessor::calculate_Pvalue(char *array1, int array1_size, char *array2, int array2_size) {
		
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
