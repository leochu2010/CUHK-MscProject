#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

int main(int argc, char* argv[]) {
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	std::cout << "number of GPU:" << deviceCount << std::endl;
}