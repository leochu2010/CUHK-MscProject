CC=/usr/local/gcc481/bin/g++
CFLAGS=-I /usr/local/cuda/include -std=c++11 -Wall
NVCC=/usr/local/cuda/bin/nvcc
NVCFLAGS=-ccbin /usr/local/gcc481/bin/g++ -c -I /usr/local/cuda/include -arch=sm_20 -std=c++11
OBJS=
cpu:
	$(CC) -c -DCOMPILE_C gpu_accel_alg/*.cpp gpu_accel_alg/utils/*.cpp feature_selection/*.cpp $(CFLAGS) 
	$(CC) -c -DCOMPILE_C gpu_accel_alg/lib/MIToolbox/*.c $(CFLAGS)
gpu:
	$(NVCC) -c gpu_accel_alg/*.cu $(NVCFLAGS) 
	rm -f getgpuinfo.o
main:
	$(CC) -o calculate *.o -L /usr/local/cuda/lib64 -lcudart -lcuda $(CFLAGS)
	ln -s calculate cal
	rm -f *.o
getgpuinfo:
	rm -f getgpuinfo 
	$(NVCC) -o getgpuinfo getgpuinfo.cu -L /usr/local/cuda/lib64 -lcudart -lcuda
clean:
	rm -f *.o
	rm -f *.gch
	rm -f calculate
	rm -f cal
all: clean cpu gpu main
