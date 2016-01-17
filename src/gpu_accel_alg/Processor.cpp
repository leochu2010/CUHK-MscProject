#include "Processor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

using namespace std;

void Processor::setDebug(bool debug){
	this->debug = debug;
}
bool Processor::isDebugEnabled(){
	return this->debug;
}

int Processor::getFeaturesPerArray(int numOfFeatures, int arrayNumbers){	
	return ceil((numOfFeatures/(float)arrayNumbers));	
}

