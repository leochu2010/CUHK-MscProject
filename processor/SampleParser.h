#ifndef _SAMPLEPARSER_H
#define _SAMPLEPARSER_H
#include "stdio.h"
#include "stdlib.h"
#include "Constant.h"

class SampleParser
{
public:
	SampleParser(char* filepath);
	
	bool retrieveSamples(int*** postiveSamples, int*** netgativeSamples, int* positiveSamplesSize, int* negativeSamplesSize);
private:
	bool OpenFile();
	void SetFilePath(char* filepath);
	bool FgetDataLine(char* data);
	bool FseekToData();
	bool FgetLine(char* data);
	bool FisDataStartLine(char* data);
	bool Parse(char* _row, int* _datas);
	bool ParseTestData(char* row, int* datas);
	char *_filePath;
	FILE *_filePointer;
};
#endif

