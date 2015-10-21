#include "SNPArffParser.h"



SNPArffParser::SNPArffParser()
{
}


SNPArffParser::~SNPArffParser()
{
}

StructArff* SNPArffParser::ParseSNPArffFile(char* filepath)
{
	fp = fopen(filepath, "r");
	if (fp == NULL)
	{
		return NULL;
	}
	StructArff* arffresult = new StructArff;
	ReadHeads(arffresult);
	ReadDatas(arffresult);


	return arffresult;
}

void SNPArffParser::DisplayArffNames(StructArff* arffstruct)
{
	for (int i = 0; i < arffstruct->FeatureCount; i++)
		DisplayArffNames(arffstruct, i);
}

void SNPArffParser::DisplayArffData(StructArff* arffstruct)
{
	for (int i = 0; i < arffstruct->SampleCount; i++)
		DisplayArffData(arffstruct, i);
}

void SNPArffParser::DisplayArffLabels(StructArff* arffstruct)
{
	for (int i = 0; i < arffstruct->FeatureCount; i++)
		DisplayArffLabels(arffstruct, i);
}

void SNPArffParser::DisplayArffNames(StructArff* arffstruct, int index)
{
	if (index >= arffstruct->FeatureCount)
		return;
	printf("%d: %s \t", index, arffstruct->SNPNames[index]);
}

void SNPArffParser::DisplayArffData(StructArff* arffstruct, int index)
{
	if (index >= arffstruct->SampleCount)
		return;
	printf("%d: \t", index);
	for (int i = 0; i < arffstruct->FeatureCount; i++)
		printf("%d\t", arffstruct->Matrix[index*arffstruct->FeatureCount + i]);
	printf("\n");
}

void SNPArffParser::DisplayArffLabels(StructArff* arffstruct, int index)
{
	if (index >= arffstruct->FeatureCount)
		return;
	printf("%d: %d \t", index, arffstruct->Labels[index]);
}

char* SNPArffParser::ReadDataLine(int featurecount)
{
	if (feof(fp))
	{
		return NULL;
	}
	char* line = new char[2 * featurecount + 1];
	if (fread(line, sizeof(char), 2 * featurecount + 1, fp) < 2 * featurecount + 1)
	{
		return NULL;
	}
	char c = fgetc(fp);
	while (c!='\r'&&c!='\n')
	{
		c = fgetc(fp);
	}
	c = fgetc(fp);
	if (c != '\r' && c != '\n')
		fseek(fp, -1, SEEK_CUR);
	return line;
}

char* SNPArffParser::ReadHeadLine()
{
	int i = 0;
	char* line = new char[100];
	if (fp == NULL || feof(fp))
	{
		return NULL;
	}
	char buffer = 0;
	while ((buffer = getc(fp)) != EOF)
	{
		line[i] = buffer;
		i++;
		if (buffer == 10)
		{
			line[i - 1] = 0;
			return line;
		}
	}
	return NULL;
}

void SNPArffParser::ReadHeads(StructArff* arffstruct)
{
	arffstruct->FeatureCount = 0;
	CharList* lastCharlist = NULL;
	char* line = ReadHeadLine();
	while (!StartWith(line, "@DATA"))
	{
		CharList* name = ExtractName(line);
		if (name == NULL)
		{
			arffstruct->FeatureCount *= -1;
			return;
		}
		if (StartWith(line, "@ATTRIBUTE"))
		{
			arffstruct->FeatureCount++;
			name->next = lastCharlist;
			lastCharlist = name;
		}
		else
		{
			if (name->data!=NULL)
			{
				delete(name->data);
			}
			delete(name);
		}
		line = ReadHeadLine();
	}

	CharList* tempname;
	tempname = lastCharlist;
	lastCharlist = lastCharlist->next;
	delete(tempname->data);
	delete(tempname);

	arffstruct->FeatureCount--;

	arffstruct->SNPNames = new char*[arffstruct->FeatureCount];
	
	for (int i = arffstruct->FeatureCount - 1; i >= 0; i--)
	{
		arffstruct->SNPNames[i] = new char[strlen(lastCharlist->data) + 1];
		memcpy(arffstruct->SNPNames[i], lastCharlist->data, strlen(lastCharlist->data) + 1);
		tempname = lastCharlist;
		lastCharlist = lastCharlist->next;
		delete(tempname->data);
		delete(tempname);
	}
}

void SNPArffParser::ReadDatas(StructArff* arffstruct)
{
	arffstruct->SampleCount = 0;
	CharList* lastline = NULL;
	char* line = ReadDataLine(arffstruct->FeatureCount);
	while (line != NULL)
	{
		arffstruct->SampleCount++;
		CharList* templine = new CharList;
		templine->data = line;
		templine->next = lastline;
		lastline = templine;
		line = ReadDataLine(arffstruct->FeatureCount);
	}

	arffstruct->Matrix = new char[arffstruct->FeatureCount*arffstruct->SampleCount];
	arffstruct->Labels = new char[arffstruct->SampleCount];
	CharList* templine;

	int sampleIndex = arffstruct->SampleCount-1;
	while (lastline != NULL)
	{
		if (sampleIndex<0)
		{
			printf("Error when read file data\n");
			return;
		}
		for (int i = 0; i < arffstruct->FeatureCount; i++)
		{
			arffstruct->Matrix[sampleIndex*arffstruct->FeatureCount + i] = lastline->data[i * 2] - '0';
		}
		if (lastline->data[2 * arffstruct->FeatureCount] == 't')
		{
			arffstruct->Labels[sampleIndex] = 1;
		}
		else
		{
			arffstruct->Labels[sampleIndex] = 0;
		}
		sampleIndex--;
		templine = lastline;
		lastline = lastline->next;
		delete(templine->data);
		delete(templine);
	}
}

bool SNPArffParser::StartWith(char* basestring, char* headstring)
{
	if (strlen(basestring) < strlen(headstring))
		return false;
	for (int i = 0; i < strlen(headstring); i++)
	{
		if (basestring[i] != headstring[i])
			return false;
	}
	return true;
}

bool SNPArffParser::EndWith(char* basestring, char* tailstring)
{
	if (strlen(basestring) < strlen(tailstring))
		return false;
	for (int i = 0; i < strlen(tailstring); i++)
	{
		if (basestring[strlen(basestring) - 1 - i] != tailstring[strlen(tailstring) - 1 - i])
			return false;
	}
	return true;
}

CharList* SNPArffParser::ExtractName(char* line)
{
	int pos1 = -1;
	int pos2 = -1;
	int index = 0;
	while (index<strlen(line))
	{
		if (line[index] == ' ')
			break;
		index++;
	}
	if (index >= strlen(line))
	{
		CharList* name = new CharList;
		name->data = NULL;
		name->next = NULL;
		return name;
	}
	pos1 = index;
	index++;
	while (index<strlen(line))
	{
		if (line[index] == ' ')
			break;
		index++;
	}
	pos2 = index;
	if (pos1 >= 0 && pos2>pos1)
	{
		CharList* name = new CharList;
		name->data = new char[pos2 - pos1];
		memcpy(name->data, line + pos1 + 1, pos2 - pos1);
		name->data[pos2 - pos1 - 1] = 0;
		return name;
	}
	return NULL;
}
