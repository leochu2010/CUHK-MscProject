#ifndef TIMER_H
#define TIMER_H

class Timer
{
	private:
		long begin, end;	
		const char *myLabel;
	public:
       	Timer(const char *label);
		void start();
		void stop();
		long getStartTime();
		long getStopTime();
		long getTimeSpent();	
		void printTimeSpent();

};

#endif
