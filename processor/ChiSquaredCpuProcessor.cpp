#include "ChiSquaredCpuProcessor.h"
#include "Timer.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


/* Numerical integration method */
double ChiSquaredCpuProcessor::Simpson3_8(double a, double b, int N, double aa1)
{
    int j;
    double l1;
    double h = (b-a)/N;
    double h1 = h/3.0;
    double sum = f0(a,aa1) + f0(b,aa1);
 
    for (j=3*N-1; j>0; j--) {
        l1 = (j%3)? 3.0 : 2.0;
        sum += l1*f0(a+h1*j,aa1) ;
    }
    return h*sum/8.0;
}
 
#define A 12
double ChiSquaredCpuProcessor::Gamma_Spouge( double z )
{
    int k;
    static double cspace[A];
    static double *coefs = NULL;
    double accum;
    double a = A;
 
    if (!coefs) {
        double k1_factrl = 1.0;
        coefs = cspace;
        coefs[0] = sqrt(2.0*M_PI);
        for(k=1; k<A; k++) {
            coefs[k] = exp(a-k) * pow(a-k,k-0.5) / k1_factrl;
            k1_factrl *= -k;
        }
    }
 
    accum = coefs[0];
    for (k=1; k<A; k++) {
        accum += coefs[k]/(z+k);
    }
    accum *= exp(-(z+a)) * pow(z+a, z+0.5);
    return accum/z;
}

double ChiSquaredCpuProcessor::f0( double t, double aa1)
{
    return  pow(t, aa1)*exp(-t); 
}
 
double ChiSquaredCpuProcessor::GammaIncomplete_Q( double a, double x)
{
    double y, h = 1.5e-2;  /* approximate integration step size */
	double aa1;
	
 
    /* this cuts off the tail of the integration to speed things up */
    y = aa1 = a-1;
    while((f0(y,aa1) * (x-y) > 2.0e-8) && (y < x))   y += .4;
    if (y>x) y=x;
 
    return 1.0 - Simpson3_8(0, y, (int)(y/h), aa1)/Gamma_Spouge(a);
}


double ChiSquaredCpuProcessor::chi2UniformDistance( double *ds, int dslen)
{
    double expected = 0.0;
    double sum = 0.0;
    int k;
 
    for (k=0; k<dslen; k++) 
        expected += ds[k];
    expected /= k;
 
    for (k=0; k<dslen; k++) {
        double x = ds[k] - expected;
        sum += x*x;
    }
    return sum/expected;
}
 
double ChiSquaredCpuProcessor::chi2Probability( int dof, double distance)
{
    return GammaIncomplete_Q( 0.5*dof, 0.5*distance);
}
 
int ChiSquaredCpuProcessor::chiIsUniform( double *dset, int dslen, double significance)
{
    int dof = dslen -1;
    double dist = chi2UniformDistance( dset, dslen);
    return chi2Probability( dof, dist ) > significance;
}


Result* ChiSquaredCpuProcessor::calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels)
{
	Timer t1 ("Total");
	t1.start();
	
	
	Result* testResult = new Result;
	testResult->scores=new double[numOfFeatures];
	
	double labelArray[numOfSamples];	
	for(int i=0;i<numOfSamples;i++)
	{
		labelArray[i]=labels[i];
		//std::cout<<labelArray[i];
	}
	
	for(int i=0;i<numOfFeatures;i++)
	{
		if(featureMask[i] != true){
			continue;
		}
								
		double featureArray[numOfSamples];
		
		for(int j=0; j<numOfSamples; j++)
		{
			int index = j*numOfFeatures + i;
			featureArray[j] = sampleTimesFeature[index];
			//std::cout<<featureArray[j];
		}
		
		double *dsets[] = { featureArray, labelArray};
		int dslens[] ={numOfSamples, numOfSamples};
		
		
				
		double score = 0;
		//double score = this->calculate_Pvalue(label0Array, label0Size, label1Array, label1Size);
		testResult->scores[i]=score;
		//std::cout<<"Feature "<<i<<":"<<score<<std::endl;	
		break;	
	}
	
	t1.stop();
	t1.printTimeSpent();
	

	testResult->startTime=t1.getStartTime();
	testResult->endTime=t1.getStopTime();
	return testResult;
}
