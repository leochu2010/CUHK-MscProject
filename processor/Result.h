#ifndef RESULT_H
#define RESULT_H

struct Result
{
	long startTime;
	
	long endTime;
		
	double* scores;
	
	bool success;
	
	int errorCode;
	
};

#endif