#ifndef RESULT_H
#define RESULT_H

#include <string>

using namespace std;

struct Result
{
	long startTime;
	
	long endTime;
		
	double* scores;
	
	bool success;
	
	int errorCode;
	
	string errorMessage;
	
};

#endif
