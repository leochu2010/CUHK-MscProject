#include "Timer.h"
#include <stdio.h>
#include <math.h>
#include <ctime>
#include <locale.h>
#include <sys/time.h>  

Timer::Timer(const char *label){
	myLabel = label;
}

void Timer::start(){
	struct timeval tv;

	gettimeofday(&tv, NULL);

	unsigned long long millisecondsSinceEpoch =
		(unsigned long long)(tv.tv_sec) * 1000 +
		(unsigned long long)(tv.tv_usec) / 1000;

	
	begin = millisecondsSinceEpoch;	
}

void Timer::stop(){
	struct timeval tv;

	gettimeofday(&tv, NULL);

	unsigned long long millisecondsSinceEpoch =
		(unsigned long long)(tv.tv_sec) * 1000 +
		(unsigned long long)(tv.tv_usec) / 1000;
		
	end = millisecondsSinceEpoch;
}

long Timer::getStartTime(){
	return begin;
}

long Timer::getStopTime(){
	return end;
}

long Timer::getTimeSpent(){	
	return end - begin;
}

void Timer::printTimeSinceStart(){
	struct timeval tv;

	gettimeofday(&tv, NULL);

	unsigned long long millisecondsSinceEpoch =
		(unsigned long long)(tv.tv_sec) * 1000 +
		(unsigned long long)(tv.tv_usec) / 1000;
	
	long sinceStart = millisecondsSinceEpoch - begin;
	setlocale(LC_NUMERIC, "");	
	if(sinceStart == 0){
		printf("%s spent 0 ms\n",myLabel);
	}else{
		printf("%s spent %'.d ms since timer start\n",myLabel, sinceStart);	
	}
}

void Timer::printTimeSpent(){
	setlocale(LC_NUMERIC, "");
	if(getTimeSpent() == 0){
		printf("%s spent 0 ms\n",myLabel);
	}else{
		printf("%s spent %'.d ms\n",myLabel, getTimeSpent());
	}
}
	