#ifndef _STRUCTARFF_H
#define _STRUCTARFF_H

struct StructArff
{
	int SampleCount;
	int FeatureCount;
	char* Matrix;
	char* Labels;
	char** SNPNames;
};

struct CharList
{
	char* data;
	CharList* next;
};
#endif
