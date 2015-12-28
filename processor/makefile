CC=/usr/local/gcc481/bin/g++
CFLAGS=-I /usr/local/cuda/include
NVCC=/usr/local/cuda/bin/nvcc
NVCFLAGS=-ccbin /usr/local/gcc481/bin/g++ -c -I /usr/local/cuda/include -arch=sm_20
OBJS=VectorAdd.o
cpu:
	$(CC) -c -DCOMPILE_C *.cpp
	$(CC) -c -DCOMPILE_C *.c
gpu:
	$(NVCC) -c *.cu 
	rm -f getgpuinfo.o
calculate:
	$(CC) -o calculate *.o -L /usr/local/cuda/lib64 -lcudart -lcuda
	$(CC) -o cal *.o -L /usr/local/cuda/lib64 -lcudart -lcuda
getgpuinfo:
	rm -f getgpuinfo 
	$(NVCC) -o getgpuinfo getgpuinfo.cu -L /usr/local/cuda/lib64 -lcudart -lcuda
clean:
	rm -f *.o
	rm -f *.gch
	rm -f calculate
	rm -f cal
all: clean cpu gpu calculate
