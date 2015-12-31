CC=/usr/local/gcc481/bin/g++
CFLAGS=-I /usr/local/cuda/include -std=c++11 -Wall
NVCC=/usr/local/cuda/bin/nvcc
NVCFLAGS=-ccbin /usr/local/gcc481/bin/g++ -c -I /usr/local/cuda/include -arch=sm_20
OBJS=
cpu:
	$(CC) -c -DCOMPILE_C gpu_accel/*.cpp feature_selection/*.cpp
	$(CC) -c -DCOMPILE_C gpu_accel/*.c
gpu:
	$(NVCC) -c gpu_accel/*.cu 
	rm -f getgpuinfo.o
calculate:
	$(CC) -o calculate *.o -L /usr/local/cuda/lib64 -lcudart -lcuda -lboost_system
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
all: clean cpu gpu calculate
