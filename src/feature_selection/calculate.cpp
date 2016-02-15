#include "gpu_accel_alg/SimpleMutualInformationProcessor.h"
#include "gpu_accel_alg/GpuAcceleratedMutualInformationProcessor.h"
#include "gpu_accel_alg/SimplePValueProcessor.h"
#include "gpu_accel_alg/GpuAcceleratedPValueProcessor.h"
#include "gpu_accel_alg/SimpleTTestProcessor.h"
#include "gpu_accel_alg/GpuAcceleratedTTestProcessor.h"
#include "gpu_accel_alg/SimpleProcessor.h"
#include "gpu_accel_alg/Processor.h"
#include "gpu_accel_alg/GpuAcceleratedProcessor.h"
#include "gpu_accel_alg/SimpleReliefFProcessor.h"
#include "gpu_accel_alg/GpuAcceleratedReliefFProcessor.h"
#include "Constant.h"
#include <iostream>
#include "SNPArffParser.h"
#include "StructArff.h"
#include <algorithm>
#include <fstream>
#include <string> 
#include <ctime>
#include <sys/time.h>  
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>
#include "tclap/CmdLine.h"

using namespace TCLAP;
using namespace std;

const string T_Test = "t_test";
const string P_Value = "pvalue";
const string MutualInformation = "mutual_info";
const string Relieff = "relieff";
const string TunedRelieff = "tuned_relieff";

struct ProcessorCommand{
	string algorithm;
	bool gpuAcceleration;
	int cpuCore;
	int gpuDevice;
	int gpuBlockThread;
	int gpuDeviceStream;
	bool enableGpuAccelerationThreadPool;
	bool debug;
	int relieffKNearest;
};

struct InputCommand{
	string filePath;
	string folderPath;
	int numOfTest;
	int features;
	int multiplyFeatures;
	int multiplySamples;
};

struct OutputCommand{
	bool stdout;
	string snpOrder;
	string format;
	string outputFile;
	bool displayProcessingTime;
};

struct Command{
	ProcessorCommand processorCommand;
	InputCommand inputCommand;
	OutputCommand outputCommand;
};

int getRank(float score, float* scoresAcsSorted,int numOfFeatures);

void processFile(Command command);

void processFolder(Command command);

void exportResult(StructArff* arff, Result* result, Command command);

void exportPerformance(long processingTime[], int testNum, string name, ofstream& output, OutputCommand outputCommand);

void exportAvgPerformance(long processingTime[], int testNum, string name, ofstream& output, bool stdout);

Processor* getProcessor(ProcessorCommand processorCommand);


Command parseCommand(int argc, char** argv){
	// Wrap everything in a try block.  Do this every time,
	// because exceptions will be thrown for problems.
	try {

		// Define the command line object.		
		CmdLine cmd("\"GPU Accelerated Feature Selection\" is powered by \"GPU Accelerated Data Mining Algorithm Library\". The library can boost the performance over 1000 times!. Source code is available at <https://github.com/leochu2010/CUHK-MscProject/>", ' ', "0.9");
		
		vector<string> algorithmArgAllowed;
		algorithmArgAllowed.push_back(P_Value);
		algorithmArgAllowed.push_back(MutualInformation);
		algorithmArgAllowed.push_back(Relieff);
		algorithmArgAllowed.push_back(T_Test);	
		algorithmArgAllowed.push_back(TunedRelieff);
		ValuesConstraint<string> algorithmArgAllowedVals( algorithmArgAllowed );
		ValueArg<string> algorithmArg("a","algorithm","Data Mining Algorithm",true,"default",&algorithmArgAllowedVals);
		cmd.add(algorithmArg);
		
		ValueArg<int> relieffKNearestArg("","k_nearest","K-Nearest instance for Relieff algorithm",false,5,"int");
		cmd.add(relieffKNearestArg);
		
		ValueArg<int> cpuCoreArg("c","core","Number of CPU Core",false,0,"int");
		cmd.add(cpuCoreArg);
		
		ValueArg<int> gpuDeviceArg("d","device","Number of GPU Devices",false,0,"int");
		cmd.add(gpuDeviceArg);
		ValueArg<int> gpuBlockThreadArg("t","thread","Number of threads per block (for GPU acceleration)",false,1,"int");
		cmd.add(gpuBlockThreadArg);
		ValueArg<int> gpuDeviceStreamArg("s","stream","Number of streams per device (for GPU acceleration)",false,1,"int");
		cmd.add(gpuDeviceStreamArg);
			
		ValueArg<int> numOfTestArg("","test","Number of test to be performed",false,0,"int");
		cmd.add(numOfTestArg);
		
		ValueArg<int> multiplyFeaturesArg("","multiply_features","Increase data size by multiplying the number of features",false,1,"int");
		cmd.add(multiplyFeaturesArg);
		
		ValueArg<int> multiplySamplesArg("","multiply_samples","Increase data size by multiplying the number of samples",false,1,"int");
		cmd.add(multiplySamplesArg);
				
		ValueArg<int> featuresArg("","feature","Number of features",false,0,"int");
		cmd.add(featuresArg);
		
		vector<string> snpOrderArgAllowed;		
		snpOrderArgAllowed.push_back("no");
		snpOrderArgAllowed.push_back("desc");
		snpOrderArgAllowed.push_back("asc");
		ValuesConstraint<string> snpOrderArgAllowedVals( snpOrderArgAllowed );
		ValueArg<string> snpOrderArg("","snporder","SNP score ordering",false,"asc",&snpOrderArgAllowedVals);
		cmd.add(snpOrderArg);
		
		SwitchArg gpuAccelerationArg("g","gpu","Enable GPU Acceleration",false);
		cmd.add(gpuAccelerationArg);
		
		SwitchArg enableGpuAccelerationThreadPoolArg("p","threadpool","Enable running GPU Accelerated algorithms on multiple devices in threads",false);
		cmd.add(enableGpuAccelerationThreadPoolArg);
		
		SwitchArg debugArg("","debug","Show debug message",false);
		cmd.add(debugArg);
			
		ValueArg<string> inputFileArg("","file","Input file",true,"","string");			
		ValueArg<string> inputFolderArg("","folder","Input folder",true,"","string");
		cmd.xorAdd(inputFileArg,inputFolderArg);
		
		SwitchArg stdoutArg("","stdout","Print result to screen",false);			
		cmd.add(stdoutArg);	
		
		SwitchArg displayProcessingTimeArg("","display_processing_time","Display processing time",false);			
		cmd.add(displayProcessingTimeArg);			
		
		ValueArg<string> outputFileArg("","output","Output file",false,"","string");
		cmd.add(outputFileArg);						
			
		vector<string> outputFormatArgAllowed;
		outputFormatArgAllowed.push_back("matlab");
		outputFormatArgAllowed.push_back("tab_delimited");			
		ValuesConstraint<string> outputFormatArgAllowedVals( outputFormatArgAllowed );
		ValueArg<string> outputFormatArg("","format","Output format",false,"tab_delimited",&outputFormatArgAllowedVals);
		cmd.add(outputFormatArg);	
		
		cmd.parse(argc,argv);
		
		Command command;
		
		command.processorCommand.algorithm = algorithmArg.getValue();
		command.processorCommand.gpuAcceleration = gpuAccelerationArg.getValue();
		command.processorCommand.cpuCore = cpuCoreArg.getValue();
		command.processorCommand.gpuDevice = gpuDeviceArg.getValue();
		command.processorCommand.gpuBlockThread = gpuBlockThreadArg.getValue();
		command.processorCommand.gpuDeviceStream = gpuDeviceStreamArg.getValue();
		command.processorCommand.enableGpuAccelerationThreadPool = enableGpuAccelerationThreadPoolArg.getValue();
		command.processorCommand.debug = debugArg.getValue();
		command.processorCommand.relieffKNearest = relieffKNearestArg.getValue();
		
		command.inputCommand.filePath = inputFileArg.getValue();
		command.inputCommand.folderPath = inputFolderArg.getValue();
		command.inputCommand.features = featuresArg.getValue();
		command.inputCommand.numOfTest = numOfTestArg.getValue();
		command.inputCommand.multiplyFeatures = multiplyFeaturesArg.getValue();
		command.inputCommand.multiplySamples = multiplySamplesArg.getValue();
		
		command.outputCommand.stdout = stdoutArg.getValue();
		command.outputCommand.format = outputFormatArg.getValue();
		command.outputCommand.snpOrder = snpOrderArg.getValue();
		command.outputCommand.outputFile = outputFileArg.getValue();
		command.outputCommand.displayProcessingTime = displayProcessingTimeArg.getValue();		
		return command;
	

	} catch (ArgException& e)  // catch any exceptions
	{ 
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl; 
		exit(EXIT_FAILURE);
	}	
}

int main(int argc, char ** argv) {

	Command command = parseCommand(argc, argv);
	
	if(command.inputCommand.filePath!=""){
		processFile(command);
	}else if(command.inputCommand.folderPath!=""){		
		processFolder(command);
	}else{		
		command.inputCommand.folderPath="/research/ksleung/pschu/data/snp1000";
		processFolder(command);		
	}
	
}

GpuAcceleratedProcessor* getGpuAcceleratedProcessor(ProcessorCommand processorCommand){
	
	string algorithm = processorCommand.algorithm;
	
	if(algorithm == P_Value){
		return new GpuAcceleratedPValueProcessor();
	}
	
	if(algorithm == T_Test){
		return new GpuAcceleratedTTestProcessor();
	}
	
	if(algorithm == MutualInformation){
		return new GpuAcceleratedMutualInformationProcessor();
	}
		
	if(algorithm == Relieff){
		int kNearest = processorCommand.relieffKNearest;
		return new GpuAcceleratedReliefFProcessor(kNearest);
	}
	
	return NULL;
}

SimpleProcessor* getSimpleProcessor(ProcessorCommand processorCommand){
	
	string algorithm = processorCommand.algorithm;
	
	if(algorithm == P_Value){
		return new SimplePValueProcessor();
	}
	
	if(algorithm == T_Test){
		return new SimpleTTestProcessor();
	}
	
	if(algorithm == MutualInformation){
		return new SimpleMutualInformationProcessor();
	}
	
	if(algorithm == Relieff){
		int kNearest = processorCommand.relieffKNearest;
		return new SimpleReliefFProcessor(kNearest);
	}
	
	return NULL;
}

Processor* getProcessor(ProcessorCommand processorCommand)
{		
	if(processorCommand.gpuAcceleration){
		GpuAcceleratedProcessor* gpuAcceleratedProcessor = getGpuAcceleratedProcessor(processorCommand);
		gpuAcceleratedProcessor->setNumberOfThreadsPerBlock(processorCommand.gpuBlockThread);
		gpuAcceleratedProcessor->setNumberOfDevice(processorCommand.gpuDevice);
		gpuAcceleratedProcessor->setNumberOfStreamsPerDevice(processorCommand.gpuDeviceStream);
		gpuAcceleratedProcessor->setDebug(processorCommand.debug);
		return gpuAcceleratedProcessor;
	}else{
		SimpleProcessor* simpleProcessor = getSimpleProcessor(processorCommand);
		simpleProcessor->setNumberOfCores(processorCommand.cpuCore);
		simpleProcessor->setDebug(processorCommand.debug);
		return simpleProcessor;
	}
	
}

void processFile(Command command){
		
	InputCommand inputCommand = command.inputCommand;
	OutputCommand outputCommand = command.outputCommand;
		
	SNPArffParser parser;
	cout << inputCommand.filePath;
	
	//read data
	StructArff* arff=parser.ParseSNPArffFile(inputCommand.filePath);	
		
	//incrase data size by multiplying features or samples
	int multiplyFeatures = inputCommand.multiplyFeatures;
	int multiplySamples = inputCommand.multiplySamples;
	
	if(multiplyFeatures >= 1 || multiplySamples >= 1){
		int newFeatureCount = arff->FeatureCount * multiplyFeatures;
		int newSampleCount = arff->SampleCount * multiplySamples;
		int oldFeatureCount = arff->FeatureCount;
		int oldSampleCount = arff->SampleCount;
		
		char* newMatrix = new char[newFeatureCount * newSampleCount];
		char* newLabels = new char[newSampleCount];
		char** newSNPNames = new char*[newFeatureCount];
		
		for(int mf=0; mf<multiplyFeatures; mf++){
			for(int f=0; f<oldFeatureCount; f++){	
				int newf = mf*oldFeatureCount + f;
				for(int ms=0; ms<multiplySamples; ms++){
					for(int s=0; s<oldSampleCount; s++){
						int newi = (ms*oldSampleCount + s) * newFeatureCount + newf;
						int oldi = s*oldFeatureCount + f;
						newMatrix[newi] = arff->Matrix[oldi];
					}
				}
				
				newSNPNames[newf]=arff->SNPNames[f];
			}			
		}		
				
		for(int ms=0; ms<multiplySamples; ms++){
			for(int s=0; s<oldSampleCount; s++){
				newLabels[ms*oldSampleCount+s] = arff->Labels[s];
			}
		}
				
		arff->SampleCount = newSampleCount;
		arff->FeatureCount = newFeatureCount;
		arff->Matrix = newMatrix;
		arff->Labels = newLabels;
		arff->SNPNames = newSNPNames;		
	}
	
	/*
	for(int i=0;i<arff->FeatureCount;i++){
		cout<<i<<":";
		for(int j=800;j<arff->SampleCount;j++){
			cout<<0+arff->Matrix[i*arff->SampleCount+j];
		}
		cout<<endl;
	}

	for(int j=0;j<arff->SampleCount;j++){		
		if(arff->Labels[j]){
			cout<<1;
		}else{
			cout<<0;
		}
	}
	cout<<endl;
	*/
	
	//decrease data size by masking unwanted features
	int features = inputCommand.features;
	if(features==0){
		features = arff->FeatureCount;
	}
	cout<<"features="<<features<<endl;
	bool featureMask[arff->FeatureCount];
	for(int i=0;i<arff->FeatureCount;i++){
		if(i<features){
			featureMask[i]=true;
		}else{
			featureMask[i]=false;
		}
	}	
	
	int testNum = inputCommand.numOfTest;
	if(testNum >0){
		
		ofstream output;
		if (!outputCommand.stdout){
			output.open(outputCommand.outputFile.c_str());
			output<<"means_ms=[];\n";		
		}
		
		long processingTime[testNum];
		
		Processor* myProcessor = getProcessor(command.processorCommand);
				
		for(int i=0;i<testNum;i++){
			Result* r = myProcessor->calculate(arff->SampleCount, arff->FeatureCount, arff->Matrix, featureMask, arff->Labels);			
			processingTime[i] = r->endTime - r->startTime;
		}		
		exportPerformance(processingTime, testNum, "test", output, outputCommand);
		
		if (!outputCommand.stdout){
			output.close();
		}
		
	} else {
		Processor* myProcessor = getProcessor(command.processorCommand);
		Result* result = myProcessor->calculate(arff->SampleCount, arff->FeatureCount, arff->Matrix, featureMask, arff->Labels);		
		//cout<<result->success<<endl;
		if (!result->success){
			cout<<result->errorMessage<<endl;
			return;
		}
		exportResult(arff, result, command);
	}
}

void exportPerformance(long processingTime[], int testNum, string name, ofstream& output, OutputCommand outputCommand){
	
	bool stdout = outputCommand.stdout;
	bool displayProcessingTime = outputCommand.displayProcessingTime;
	
	//cout<<"init:"<<processingTime[0]<<" ms"<<endl;	
	if (!displayProcessingTime){
		if (!stdout){
			//output<<name<<"_init_ms="<<processingTime[0]<<";\n";
			output<<name<<"_processing_ms=[";
		}else if (stdout){
			//cout<<name<<"_init_ms="<<processingTime[0]<<";\n";
			cout<<name<<"_processing_ms=[";
		}
	}
	
	float total=0;
	for(int i=0; i<testNum ;i++){
		if (!stdout){
			output<<" "<<processingTime[i];		
		}
		total+=processingTime[i];
	}
	float mean = (total/testNum);
	if (!stdout){
		output<<"];\n";	
		output<<"means_ms=[means_ms "<<mean<<"];\n";
	}else if(displayProcessingTime){
		cout<< mean;
	}else if (stdout){
		cout<<"];\n";	
		cout<<"means_ms=[means_ms "<<mean<<"];\n";
	}
}

void exportAvgPerformance(long processingTime[], int testNum, string name, ofstream& output, bool stdout){
			
	//cout<<"init:"<<processingTime[0]<<" ms"<<endl;
	
	float total=0;
	if(!stdout){
		output<<"#";
	}else{	
		cout<<"#";
	}
	for(int i=1; i<testNum ;i++){
		if(!stdout){
			output<<" "<<processingTime[i];		
		}else{
			cout<<" "<<processingTime[i];
		}
		total+=processingTime[i];
	}
	float mean = total/(testNum-1);		
	if (!stdout){
		output<<"\n";	
		output<<"means_ms=[means_ms "<<mean<<"];\n";
	}else{
		cout<<"\n";
		cout<<"means_ms=[means_ms "<<mean<<"];\n";
	}
}

void exportResult(StructArff* arff, Result* r, Command command)
{
	OutputCommand outputCommand = command.outputCommand;
	if(outputCommand.format == "tab_delimited"){
		cout<<"output format="<<outputCommand.format<<endl;
		ofstream output;
		output.open(outputCommand.outputFile.c_str());
		
		output << "@" << outputCommand.outputFile <<"\n";
				
		multimap<float,string> scoreFeatureMap;
		
		if(outputCommand.snpOrder == "asc"){
			for(int i=0; i<arff->FeatureCount; i++){
				scoreFeatureMap.insert(pair<float, string>(r->scores[i], string(arff->SNPNames[i])));
			}
			
			int rank = 1;
			for(multimap<float,string>::iterator it = scoreFeatureMap.begin(); it!= scoreFeatureMap.end(); ++it){	
				output<<(*it).second <<"\t"<<(*it).first<<"\t"<<rank<<endl;
				rank+=1;
			} 
		}else if(outputCommand.snpOrder == "desc"){
			for(int i=0; i<arff->FeatureCount; i++){
				scoreFeatureMap.insert(pair<float, string>(r->scores[i], string(arff->SNPNames[i])));
			}
			
			int rank = 1;
			for(multimap<float,string>::reverse_iterator it = scoreFeatureMap.rbegin(); it!= scoreFeatureMap.rend(); ++it){	
				output<<(*it).second <<"\t"<<(*it).first<<"\t"<<rank<<endl;
				rank+=1;
			} 			
		}else if(outputCommand.snpOrder == "no"){
			for(int i=0; i<arff->FeatureCount; i++){
				output<<string(arff->SNPNames[i]) <<"\t"<<r->scores[i]<<endl;
			}
		}
		
		output.close();
		
	}
	
	if(outputCommand.format == "matlab"){		
		ofstream rankFile;
		ofstream scoreFile;
		
		struct timeval tv;
		gettimeofday(&tv, NULL);
		unsigned long long millisecondsSinceEpoch =
			(unsigned long long)(tv.tv_sec) * 1000 +
			(unsigned long long)(tv.tv_usec) / 1000;
			
		stringstream strstream0;
		stringstream strstream1;
		string rankFilename;
		string scoreFilename;
		
		
		strstream0 << "./output/" << command.processorCommand.algorithm << "_ranks_" << millisecondsSinceEpoch << ".txt";
		strstream1 << "./output/" << command.processorCommand.algorithm << "_scores_" << millisecondsSinceEpoch << ".txt";
		strstream0>>rankFilename;
		strstream1>>scoreFilename;
		
		rankFile.open (rankFilename.c_str());
		scoreFile.open (scoreFilename.c_str());
		
		float sortScores[arff->FeatureCount];
		for(int s=0;s<arff->FeatureCount;s++)
		{
			sortScores[s]=r->scores[s];
			scoreFile << r->scores[s] << " ";
			//cout<<sortScores[s]<<" ";
		}
		
		sort(sortScores,sortScores+arff->FeatureCount);

		int p1Ranks;
		int p2Ranks;
						
		scoreFile << "filepath="<<outputCommand.outputFile <<": scores=[";
				
		float p1Score = r->scores[(arff->FeatureCount-2)];
		float p2Score = r->scores[(arff->FeatureCount-1)];
		p1Ranks = getRank(p1Score, sortScores, arff->FeatureCount);
		p2Ranks = getRank(p2Score, sortScores, arff->FeatureCount);
					
		cout<<endl<<"P1 Score:"<<p1Score<<" P2 Score:"<<p2Score<<endl;
		
		//cout<<endl<<endl;
		//cout<<"P1 Score="<<p1Score<<", Rank="<<p1Rank<<endl;
		//cout<<"P2 Scire="<<p2Score<<", Rank="<<p2Rank<<endl;			
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
		
		cout<<endl<<"expoted the result to: "<<endl<<scoreFilename <<endl<<rankFilename<<endl;
	}
	
}

void processFolder(Command command)
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

	
	ofstream rankFile;
	ofstream scoreFile;
	
	struct timeval tv;
	gettimeofday(&tv, NULL);
	unsigned long long millisecondsSinceEpoch =
		(unsigned long long)(tv.tv_sec) * 1000 +
		(unsigned long long)(tv.tv_usec) / 1000;
		
	stringstream strstream0;
	stringstream strstream1;
	string rankFilename;
	string scoreFilename;
	
	
	strstream0 << "./output/" << command.processorCommand.algorithm << "_ranks_" << millisecondsSinceEpoch << ".txt";
	strstream1 << "./output/" << command.processorCommand.algorithm << "_scores_" << millisecondsSinceEpoch << ".txt";
	strstream0>>rankFilename;
	strstream1>>scoreFilename;
	
	rankFile.open (rankFilename.c_str());
	scoreFile.open (scoreFilename.c_str());
	
	Processor* myProcessor = getProcessor(command.processorCommand);
	
	
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
			
			memset(filepath, 0, 200);
			strcat(filepath, command.inputCommand.folderPath.c_str());
			strcat(filepath, tempfolder);
			strcat(filepath, tempfile);
			
			cout << filepath;
			StructArff* arff=parser.ParseSNPArffFile(filepath);
			
			scoreFile << "filepath="<<filepath <<": scores=[";
									
			numOfFeatures = arff->FeatureCount;
			////
			//
			// you can add your code here
			////
						
			//pvalueProcessor.calculate(1, 1, 1, 1, 1);
			//virtual Result calculate(int numOfSamples, int numOfFeatures, char* sampleTimesFeature, bool* featureMask, char* label) = 0;
		
			int features = command.inputCommand.features;
			if(features==0){
				features = arff->FeatureCount;
			}
			cout<<"features="<<features<<endl;
			bool featureMask[arff->FeatureCount];
			for(int i=0;i<arff->FeatureCount;i++){
				if(i<features){
					featureMask[i]=true;
				}else{
					featureMask[i]=false;
				}
			}
						
			Result* r = myProcessor->calculate(arff->SampleCount, arff->FeatureCount, arff->Matrix, featureMask, arff->Labels);
			if (!r->success){
				cout<<r->errorMessage<<endl;
				return;
			}
			
			
			float sortScores[arff->FeatureCount];
			for(int s=0;s<arff->FeatureCount;s++)
			{
				
				sortScores[s]=r->scores[s];
				scoreFile << r->scores[s] << " ";
				//cout<<sortScores[s]<<" ";
			}

			
			sort(sortScores,sortScores+arff->FeatureCount);
		
			float p1Score = r->scores[(arff->FeatureCount-2)];
			float p2Score = r->scores[(arff->FeatureCount-1)];
			p1Ranks[fileindex] = getRank(p1Score, sortScores, arff->FeatureCount);
			p2Ranks[fileindex] = getRank(p2Score, sortScores, arff->FeatureCount);
						
			cout<<endl<<"P1 Score:"<<p1Score<<" P2 Score:"<<p2Score<<endl;
			
			//cout<<endl<<endl;
			//cout<<"P1 Score="<<p1Score<<", Rank="<<p1Rank<<endl;
			//cout<<"P2 Scire="<<p2Score<<", Rank="<<p2Rank<<endl;			
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
	return -1;
}