#ifndef CHISQUAREDCPUPROCESSOR_H
#define CHISQUAREDCPUPROCESSOR_H

#include "CpuProcessor.h"
class ChiSquaredCpuProcessor : public CpuProcessor 
{
	private:		
		double Simpson3_8(double a, double b, int N, double aa1);
		double Gamma_Spouge( double z );		
		double f0( double t, double aa1);
		double GammaIncomplete_Q( double a, double x);
		double chi2UniformDistance( double *ds, int dslen);
		double chi2Probability( int dof, double distance);
		int chiIsUniform( double *dset, int dslen, double significance);
	public:
        	virtual Result* calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* labels);

};

#endif
