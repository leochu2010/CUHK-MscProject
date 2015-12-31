#ifndef _SNPARFFPARSER_H
#define _SNPARFFPARSER_H
#include "StructArff.h"
#include "stdio.h"
#include "string.h"
class SNPArffParser
{
public:
	SNPArffParser();
	~SNPArffParser();
	StructArff* ParseSNPArffFile(char* filepath);

	void DisplayArffNames(StructArff* arffstruct);
	void DisplayArffData(StructArff* arffstruct);
	void DisplayArffLabels(StructArff* arffstruct);
	void DisplayArffNames(StructArff* arffstruct, int index);
	void DisplayArffData(StructArff* arffstruct, int index);
	void DisplayArffLabels(StructArff* arffstruct, int index);
private:
	char* ReadDataLine(int feauturecount);
	char* ReadHeadLine();
	void ReadHeads(StructArff* arffstruct);
	void ReadDatas(StructArff* arffstruct);
	bool StartWith(char* basestring, char* headstring);
	bool EndWith(char* basestring, char* tailstring);
	CharList* ExtractName(char* line);
	FILE *fp;
};

#endif


