#include "MutualInformationCpuProcessor.h"
#include "PValueCpuProcessor.h"
#include "CpuProcessor.h"
#include "time.h"
#include "Constant.h"
#include <iostream>
#include "SNPArffParser.h"
#include "StructArff.h"
#include <algorithm>
#include <fstream>
#include <string> 

int getRank(float score, float* scoresAcsSorted,int numOfFeatures);

void process(char* folderbase, std::string processor);

CpuProcessor* getCpuProcessor(std::string processor);

int main(int argc, char* argv[]) {

	std::string processor;

	if(argc < 2)
	{
		std::cout << "Usage is -p <processor> \n"; // Inform the user of how to use the program		
		exit(0);
	} 
		
	for (int i = 1; i < argc; i++) {		
		if (i + 1 != argc) // Check that we haven't finished parsing already			
			if (std::string(argv[i]) == "-p") {				
				processor = argv[i + 1];
				
				if(processor != "pvalue" && processor != "mutual_info"){
					std::cout << "Invalue process, please try any one of them (pvalue, mutual_info).\n";						
					exit(0);
				}
			} else {
				std::cout << "Not enough or invalid arguments, please try again.\n";                    
				exit(0);
		}            
	}
	
	
	char folderbase[100];
	memset(folderbase, 0, 100);
	strcat(folderbase, "/uac/msc/pschu/project/Algorithm/data/snp1000");	
	process(folderbase, processor);

	return 0;
}

CpuProcessor* getCpuProcessor(std::string processor)
{
	if (processor == "mutual_info"){
		return new MutualInformationCpuProcessor();
	}else if(processor == "pvalue"){
		return new PValueCpuProcessor();
	}
}

void process(char* folderbase, std::string processor)
{
	char* filepath = new char[100];

	char folder[5];
	char tempfolder[5];
	memset(folder, 0, 5);
	strcat(folder, "/h01");
	folder[4] = 0;

	char file[21];
	char tempfile[21];
	memset(file, 0, 21);
	strcat(file, "/01_EDM-1_01txt.arff");
	file[20] = 0;

	
	std::ofstream myfile;
	myfile.open ("ranks.txt");
	
	
	
	SNPArffParser parser;
	for (int folderindex = 0; folderindex < 4; folderindex++)
	{
		strcpy(tempfolder, folder);
		tempfolder[3] += folderindex;
		
		int p1Ranks[100];
		int p2Ranks[100];

		for (int fileindex = 0; fileindex < 100; fileindex++)
		{
			strcpy(tempfile, file);
			tempfile[2] += folderindex;
			tempfile[8] += fileindex / 50;
			tempfile[10] += fileindex % 50 / 10;
			if (fileindex % 50 % 10<9)
			{
				tempfile[11] += fileindex % 50 % 10;
			}
			else
			{
				tempfile[10] += 1;
				tempfile[11] = '0';
			}
			
			memset(filepath, 0, 100);
			strcat(filepath, folderbase);
			strcat(filepath, tempfolder);
			strcat(filepath, tempfile);
			
			std::cout << filepath;
			StructArff* arff=parser.ParseSNPArffFile(filepath);
									
			////
			//
			// you can add your code here
			////
			CpuProcessor* cpuProcessor = getCpuProcessor(processor);			
			
			//pvalueProcessor.calculate(1, 1, 1, 1, 1);
			//virtual Result calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* label) = 0;

			bool featureMask[arff->FeatureCount];
			for(int i=0;i<arff->FeatureCount;i++){
				featureMask[i]=true;
			}
			Result* r = cpuProcessor->calculate(arff->SampleCount, arff->FeatureCount, arff->Matrix, featureMask, arff->Labels);
			
			
			float sortScores[arff->FeatureCount];
			for(int s=0;s<arff->FeatureCount;s++)
			{
				
				sortScores[s]=r->scores[s];
				//std::cout<<sortScores[s]<<" ";
			}

			
			//std::cout<<std::endl<<std::endl;
			std::cout<<r->startTime<<"-"<<r->endTime<<":"<<r->endTime-r->startTime<<std::endl<<std::endl;
			
			std::sort(sortScores,sortScores+arff->FeatureCount);
		
			float p1Score = r->scores[(arff->FeatureCount-2)];
			float p2Score = r->scores[(arff->FeatureCount-1)];
			p1Ranks[fileindex] = getRank(p1Score, sortScores, arff->FeatureCount);
			p2Ranks[fileindex] = getRank(p2Score, sortScores, arff->FeatureCount);
						
			std::cout<<std::endl<<"P1 Score:"<<p1Score<<" P2 Score:"<<p2Score<<std::endl;
			
			//std::cout<<std::endl<<std::endl;
			//std::cout<<"P1 Score="<<p1Score<<", Rank="<<p1Rank<<std::endl;
			//std::cout<<"P2 Scire="<<p2Score<<", Rank="<<p2Rank<<std::endl;			
		}
		
		
		//write p1Ranks, p2Ranks to a file
		myfile << "p1=[";
		
		for(int r=0;r<100;r++)
		{
			myfile << p1Ranks[r] << " ";
		}
		myfile << "]\n";		
		
		myfile << "p2=[";
		for(int r=0;r<100;r++)
		{
			myfile << p2Ranks[r] << " ";
		}
		myfile << "]\n";		
		
	}
	myfile.close();
}

int getRank(float score, float* scoresAcsOrder, int numOfFeatures){
	for(int i=0;i<numOfFeatures;i++){
		if (score == scoresAcsOrder[i]){
			return i;
		}
	}
	
}
