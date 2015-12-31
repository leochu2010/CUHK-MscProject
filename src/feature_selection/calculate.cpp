#include "../gpu_accel/MutualInformationCpuProcessor.h"
#include "../gpu_accel/PValueCpuProcessor.h"
#include "../gpu_accel/PValueGpuProcessor.h"
#include "../gpu_accel/CpuProcessor.h"
#include "../gpu_accel/Processor.h"
#include "../gpu_accel/GpuProcessor.h"
#include "../gpu_accel/Constant.h"
#include "../gpu_accel/Timer.h"
#include <iostream>
#include "../gpu_accel/SNPArffParser.h"
#include "../gpu_accel/StructArff.h"
#include <algorithm>
#include <fstream>
#include <string> 
#include <ctime>
#include <sys/time.h>  
#include <sstream>
#include <map>

int getRank(float score, float* scoresAcsSorted,int numOfFeatures);

void processFile(char* filepath, std::string algorithm, std::string processor, std::string outputFile, std::string outputFormat, std::string device, std::string thread, std::string feature, std::string test, std::string try_thread_from, std::string try_thread_to, std::string try_thread_step, std::string stdout, std::string ordering);

void processFolder(char* folderbase, std::string algorithm, std::string processor, std::string device, std::string thread, std::string feature);

void exportResult(StructArff* arff, Result* result, char* filepath, std::string outputFile, std::string outputFormat, std::string algorithm, std::string ordering);

void exportPerformance(long processingTime[], int testNum, std::string name, std::ofstream& output, std::string stdout);

void exportAvgPerformance(long processingTime[], int testNum, std::string name, std::ofstream& output, std::string stdout);

Processor* getProcessor(std::string algorithm, std::string processor, std::string device, std::string thread, std::string feature);

int main(int argc, char* argv[]) {

	std::string processor="cpu";
	std::string algorithm;
	std::string inputFile="";
	std::string inputFolder="";
	std::string outputFile="";
	std::string outputFormat="tab_delimited";
	std::string ordering="asc";
	std::string device="0";
	std::string thread="0";
	std::string try_thread_from="0";
	std::string try_thread_to="0";
	std::string try_thread_step="10";
	std::string feature="-1";
	std::string test="0";
	std::string stdout="0";

	if(argc < 6)
	{
		std::cout << "Usage is -algorithm <algorithm> -processor <processor> (-file <input file> | -folder <input folder>)\n"; // Inform the user of how to use the program	
		exit(0);
	} 
		
	for (int i = 1; i < argc; i++) {		
		if (i + 1 != argc) 
			if (std::string(argv[i]) == "-algorithm") {
				algorithm = argv[i + 1];
				
				if(algorithm != "pvalue" && algorithm != "mutual_info"){
					std::cout << "Invalue input, only accept {pvalue, mutual_info}.\n";						
					exit(0);
				}
				i +=1;
			} else if(std::string(argv[i]) == "-processor"){
				processor = argv[i + 1];
				if(processor != "cpu" && processor != "gpu"){
					std::cout << "Invalue input, only accept {cpu, gpu}.\n";						
					exit(0);
				}
				i +=1;
			} else if(std::string(argv[i]) == "-device"){
				device = argv[i + 1];
				i +=1;
			} else if(std::string(argv[i]) == "-thread"){
				thread = argv[i + 1];
				i +=1;
			} else if(std::string(argv[i]) == "-feature"){
				feature = argv[i + 1];
				i +=1;
			} else if(std::string(argv[i]) == "-file"){
				inputFile = argv[i + 1];				
				i +=1;
			} else if(std::string(argv[i]) == "-output"){
				outputFile = argv[i + 1];
				i +=1;	
			} else if(std::string(argv[i]) == "-ordering"){
				ordering = argv[i + 1];
				i +=1;
			} else if(std::string(argv[i]) == "-stdout"){
				stdout = argv[i + 1];
				i +=1;	
			} else if(std::string(argv[i]) == "-test"){
				test = argv[i + 1];
				i +=1;				
			} else if(std::string(argv[i]) == "-try_thread_to"){
				try_thread_to = argv[i + 1];
				i +=1;							
			} else if(std::string(argv[i]) == "-try_thread_from"){
				try_thread_from = argv[i + 1];
				i +=1;										
			} else if(std::string(argv[i]) == "-try_thread_step"){
				try_thread_step = argv[i + 1];
				i +=1;										
			}  else if(std::string(argv[i]) == "-format"){
				outputFormat = argv[i + 1];				
				if(outputFormat != "matlab" && outputFormat != "tab_delimited"){
					std::cout << "Invalue input, only accept {matlab, tab_delimited}.\n";						
					exit(0);
				}
				i +=1;
			} else if(std::string(argv[i]) == "-folder"){
				inputFolder = argv[i + 1];
				i +=1;				
			} else {
				std::cout << "Not enough or invalid arguments, please try again.\n";                    
				exit(0);
		}            
	}
	
	if(inputFile!=""){
		char filepath[100];
		memset(filepath,0,100);
		strcat(filepath, inputFile.c_str());
		processFile(filepath, algorithm, processor, outputFile, outputFormat, device, thread, feature, test, try_thread_from, try_thread_to, try_thread_step, stdout, ordering);
	}else if(inputFolder!=""){
		char folderbase[100];
		memset(folderbase, 0, 100);
		strcat(folderbase, inputFolder.c_str());
		processFolder(folderbase, algorithm, processor, device, thread, feature);
	}else{
		char folderbase[100];
		memset(folderbase, 0, 100);
		strcat(folderbase, "/uac/msc/pschu/project/Algorithm/data/snp1000");
		processFolder(folderbase, algorithm, processor, device, thread, feature);
	}
	

	return 0;
}

Processor* getProcessor(std::string algorithm, std::string processor, std::string device, std::string thread)
{
	//std::cout<<"algorithm="<<algorithm<<", processor="<<processor<<", device="<<device<<", thread="<<thread<<std::endl;
	if (algorithm == "mutual_info"){
		return new MutualInformationCpuProcessor();
	}else if(algorithm == "pvalue"){
		if(processor == "cpu"){
			return new PValueCpuProcessor();
		}else if(processor == "gpu"){
			GpuProcessor* gpuProcessor = new PValueGpuProcessor();
			gpuProcessor->setNumberOfThreadsPerBlock(std::atoi(thread.c_str()));
			gpuProcessor->setNumberOfDevice(std::atoi(device.c_str()));
			return gpuProcessor;
		}
	}
}

void processFile(char* filepath, std::string algorithm, std::string processor, std::string outputFile, std::string outputFormat, std::string device, std::string thread, std::string feature, std::string test, std::string try_thread_from, std::string try_thread_to, std::string try_thread_step, std::string stdout, std::string ordering){
		
	SNPArffParser parser;
	std::cout << filepath;
	
	StructArff* arff=parser.ParseSNPArffFile(filepath);	
		
	////
	//
	// you can add your code here
	////
				
	//pvalueProcessor.calculate(1, 1, 1, 1, 1);
	//virtual Result calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* label) = 0;

	int features = std::atoi(feature.c_str());
	if(features==-1){
		features = arff->FeatureCount;
	}
	std::cout<<"features="<<features<<std::endl;
	bool featureMask[arff->FeatureCount];
	for(int i=0;i<arff->FeatureCount;i++){
		if(i<features){
			featureMask[i]=true;
		}else{
			featureMask[i]=false;
		}
	}
	
	
	int testNum = std::atoi(test.c_str());
	
	
	if(testNum >0){				
					
		std::ofstream output;
		if (stdout=="0"){
			output.open(outputFile.c_str());
			output<<"means_ms=[];\n";		
		}
	
		int tryThreadTo = std::atoi(try_thread_to.c_str());
		int tryThreadFrom = std::atoi(try_thread_from.c_str());
		int tryThreadStep = std::atoi(try_thread_step.c_str());
		long processingTime[testNum];
		
		if(tryThreadTo > 0){
			if (stdout=="0"){
				output<<"threads=[];\n";
			}
			for(int t=tryThreadFrom;t<tryThreadTo;t+=tryThreadStep){
				std::stringstream threadNum;
				std::stringstream name;
				threadNum <<t;
				name << "t"<< t;
				Processor* myProcessor = getProcessor(algorithm, processor, device, threadNum.str());
				for(int i=0;i<testNum;i++){
					Result* r = myProcessor->calculate(arff->SampleCount, arff->FeatureCount, arff->Matrix, featureMask, arff->Labels);
					processingTime[i] = r->endTime - r->startTime;					
				}
				exportAvgPerformance(processingTime, testNum, name.str(), output, stdout);
				if (stdout=="0"){
					output<<"threads=[threads "<<t<<"];\n";	
				}
				
				delete myProcessor;
			}		
		}else{
			Processor* myProcessor = getProcessor(algorithm, processor, device, thread);
			for(int i=0;i<testNum;i++){
				Result* r = myProcessor->calculate(arff->SampleCount, arff->FeatureCount, arff->Matrix, featureMask, arff->Labels);
				processingTime[i] = r->endTime - r->startTime;				
			}
			exportPerformance(processingTime, testNum, "test", output, stdout);
		}
		
		if (stdout=="0"){
			output.close();
		}
		
	} else {
		Processor* myProcessor = getProcessor(algorithm, processor, device, thread);
		Result* r = myProcessor->calculate(arff->SampleCount, arff->FeatureCount, arff->Matrix, featureMask, arff->Labels);		
		exportResult(arff, r, filepath, outputFile, outputFormat, algorithm, ordering);
	}
}

void exportPerformance(long processingTime[], int testNum, std::string name, std::ofstream& output, std::string stdout){
	
		
	//std::cout<<"init:"<<processingTime[0]<<" ms"<<std::endl;	
	if (stdout=="0"){
		//output<<name<<"_init_ms="<<processingTime[0]<<";\n";
		output<<name<<"_processing_ms=[";
	}else if (stdout=="1"){
		//std::cout<<name<<"_init_ms="<<processingTime[0]<<";\n";
		std::cout<<name<<"_processing_ms=[";
	}
	
	float total=0;
	for(int i=0; i<testNum ;i++){
		if (stdout=="0"){
			output<<" "<<processingTime[i];		
		}
		total+=processingTime[i];
	}
	float mean = (total/testNum);
	if (stdout=="0"){
		output<<"];\n";	
		output<<"means_ms=[means_ms "<<mean<<"];\n";
	}else if (stdout=="1"){
		std::cout<<"];\n";	
		std::cout<<"means_ms=[means_ms "<<mean<<"];\n";
	}else if(stdout=="processing_time"){
		std::cout<< mean;
	}
}

void exportAvgPerformance(long processingTime[], int testNum, std::string name, std::ofstream& output, std::string stdout){
			
	//std::cout<<"init:"<<processingTime[0]<<" ms"<<std::endl;
	
	float total=0;
	if(stdout == "0"){
		output<<"#";
	}else{	
		std::cout<<"#";
	}
	for(int i=1; i<testNum ;i++){
		if(stdout == "0"){
			output<<" "<<processingTime[i];		
		}else{
			std::cout<<" "<<processingTime[i];
		}
		total+=processingTime[i];
	}
	float mean = total/(testNum-1);		
	if (stdout=="0"){
		output<<"\n";	
		output<<"means_ms=[means_ms "<<mean<<"];\n";
	}else{
		std::cout<<"\n";
		std::cout<<"means_ms=[means_ms "<<mean<<"];\n";
	}
}

void exportResult(StructArff* arff, Result* r, char* filepath, std::string outputFile, std::string outputFormat, std::string algorithm, std::string ordering)
{
	if(outputFormat == "tab_delimited"){
		std::cout<<"output format="<<outputFormat<<std::endl;
		std::ofstream output;
		output.open(outputFile.c_str());
		
		output << "@" << filepath <<"\n";
				
		std::multimap<float,std::string> scoreFeatureMap;
		
		if(ordering == "asc"){
			for(int i=0; i<arff->FeatureCount; i++){
				scoreFeatureMap.insert(std::pair<float, std::string>(r->scores[i], std::string(arff->SNPNames[i])));
			}
			
			int rank = 1;
			for(std::multimap<float,std::string>::iterator it = scoreFeatureMap.begin(); it!= scoreFeatureMap.end(); ++it){	
				output<<(*it).second <<"\t"<<(*it).first<<"\t"<<rank<<std::endl;
				rank+=1;
			} 
		}else if(ordering == "no"){
			for(int i=0; i<arff->FeatureCount; i++){
				output<<std::string(arff->SNPNames[i]) <<"\t"<<r->scores[i]<<std::endl;
			}
		}
		
		output.close();
		
	}
	
	if(outputFormat == "matlab"){		
		std::ofstream rankFile;
		std::ofstream scoreFile;
		
		struct timeval tv;
		gettimeofday(&tv, NULL);
		unsigned long long millisecondsSinceEpoch =
			(unsigned long long)(tv.tv_sec) * 1000 +
			(unsigned long long)(tv.tv_usec) / 1000;
			
		std::stringstream strstream0;
		std::stringstream strstream1;
		std::string rankFilename;
		std::string scoreFilename;
		
		
		strstream0 << "./output/" << algorithm << "_ranks_" << millisecondsSinceEpoch << ".txt";
		strstream1 << "./output/" << algorithm << "_scores_" << millisecondsSinceEpoch << ".txt";
		strstream0>>rankFilename;
		strstream1>>scoreFilename;
		
		rankFile.open (rankFilename.c_str());
		scoreFile.open (scoreFilename.c_str());
		
		float sortScores[arff->FeatureCount];
		for(int s=0;s<arff->FeatureCount;s++)
		{
			sortScores[s]=r->scores[s];
			scoreFile << r->scores[s] << " ";
			//std::cout<<sortScores[s]<<" ";
		}
		
		std::sort(sortScores,sortScores+arff->FeatureCount);

		int p1Ranks;
		int p2Ranks;
		int numOfFeatures;
				
		scoreFile << "filepath="<<outputFile <<": scores=[";
								
		numOfFeatures = arff->FeatureCount;
		
		float p1Score = r->scores[(arff->FeatureCount-2)];
		float p2Score = r->scores[(arff->FeatureCount-1)];
		p1Ranks = getRank(p1Score, sortScores, arff->FeatureCount);
		p2Ranks = getRank(p2Score, sortScores, arff->FeatureCount);
					
		std::cout<<std::endl<<"P1 Score:"<<p1Score<<" P2 Score:"<<p2Score<<std::endl;
		
		//std::cout<<std::endl<<std::endl;
		//std::cout<<"P1 Score="<<p1Score<<", Rank="<<p1Rank<<std::endl;
		//std::cout<<"P2 Scire="<<p2Score<<", Rank="<<p2Rank<<std::endl;			
		scoreFile << "];\n";
		//write p1Ranks, p2Ranks to a file
		rankFile << "p1=[";	
		rankFile << p1Ranks << " ";
		rankFile << "];\n";	
		
		rankFile << "p2=[";	
		rankFile << p2Ranks << " ";
		rankFile << "];\n";	
			
		rankFile.close();
		scoreFile.close();
		
		std::cout<<std::endl<<"expoted the result to: "<<std::endl<<scoreFilename <<std::endl<<rankFilename<<std::endl;
	}
	
}

void processFolder(char* folderbase, std::string algorithm, std::string processor, std::string device, std::string thread, std::string feature)
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

	
	std::ofstream rankFile;
	std::ofstream scoreFile;
	
	struct timeval tv;
	gettimeofday(&tv, NULL);
	unsigned long long millisecondsSinceEpoch =
		(unsigned long long)(tv.tv_sec) * 1000 +
		(unsigned long long)(tv.tv_usec) / 1000;
		
	std::stringstream strstream0;
	std::stringstream strstream1;
	std::string rankFilename;
	std::string scoreFilename;
	
	
	strstream0 << "./output/" << algorithm << "_ranks_" << millisecondsSinceEpoch << ".txt";
	strstream1 << "./output/" << algorithm << "_scores_" << millisecondsSinceEpoch << ".txt";
	strstream0>>rankFilename;
	strstream1>>scoreFilename;
	
	rankFile.open (rankFilename.c_str());
	scoreFile.open (scoreFilename.c_str());
	
	Processor* myProcessor = getProcessor(algorithm, processor, device, thread);
	
	
	SNPArffParser parser;
	for (int folderindex = 0; folderindex < 4; folderindex++)
	{
		strcpy(tempfolder, folder);
		tempfolder[3] += folderindex;
		
		int p1Ranks[100];
		int p2Ranks[100];
		int numOfFeatures;
		
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
			
			scoreFile << "filepath="<<filepath <<": scores=[";
									
			numOfFeatures = arff->FeatureCount;
			////
			//
			// you can add your code here
			////
						
			//pvalueProcessor.calculate(1, 1, 1, 1, 1);
			//virtual Result calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* label) = 0;
		
			int features = std::atoi(feature.c_str());
			if(features==-1){
				features = arff->FeatureCount;
			}
			std::cout<<"features="<<features<<std::endl;
			bool featureMask[arff->FeatureCount];
			for(int i=0;i<arff->FeatureCount;i++){
				if(i<features){
					featureMask[i]=true;
				}else{
					featureMask[i]=false;
				}
			}
						
			Result* r = myProcessor->calculate(arff->SampleCount, arff->FeatureCount, arff->Matrix, featureMask, arff->Labels);
			
			
			float sortScores[arff->FeatureCount];
			for(int s=0;s<arff->FeatureCount;s++)
			{
				
				sortScores[s]=r->scores[s];
				scoreFile << r->scores[s] << " ";
				//std::cout<<sortScores[s]<<" ";
			}

			
			std::sort(sortScores,sortScores+arff->FeatureCount);
		
			float p1Score = r->scores[(arff->FeatureCount-2)];
			float p2Score = r->scores[(arff->FeatureCount-1)];
			p1Ranks[fileindex] = getRank(p1Score, sortScores, arff->FeatureCount);
			p2Ranks[fileindex] = getRank(p2Score, sortScores, arff->FeatureCount);
						
			std::cout<<std::endl<<"P1 Score:"<<p1Score<<" P2 Score:"<<p2Score<<std::endl;
			
			//std::cout<<std::endl<<std::endl;
			//std::cout<<"P1 Score="<<p1Score<<", Rank="<<p1Rank<<std::endl;
			//std::cout<<"P2 Scire="<<p2Score<<", Rank="<<p2Rank<<std::endl;			
			scoreFile << "];\n";
		}
		
		
		//write p1Ranks, p2Ranks to a file
		rankFile << "p1=[";
		
		for(int r=0;r<numOfFeatures;r++)
		{
			rankFile << p1Ranks[r] << " ";
		}
		rankFile << "];\n";		
		
		rankFile << "p2=[";
		for(int r=0;r<numOfFeatures;r++)
		{
			rankFile << p2Ranks[r] << " ";
		}
		rankFile << "];\n";		
		
	}
	rankFile.close();
	scoreFile.close();
}

int getRank(float score, float* scoresAcsOrder, int numOfFeatures){
	for(int i=0;i<numOfFeatures;i++){
		if (score == scoresAcsOrder[i]){
			return i;
		}
	}
	
}
