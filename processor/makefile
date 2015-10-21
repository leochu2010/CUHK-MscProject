CC=/usr/local/gcc481/bin/g++
CFLAGS=-I /usr/local/cuda/include
NVCC=/usr/local/cuda/bin/nvcc
NVCFLAGS=-ccbin /usr/local/gcc481/bin/g++ -c -I /usr/local/cuda/include -arch=sm_20
OBJS=VectorAdd.o
VectorAddByLink: $(OBJS)
	$(CC) -o VectorAdd $(OBJS) -L /usr/local/cuda/lib64 -lcudart -lcuda
HelloWorld:
	$(CC) -o HelloWorld HelloWorld.o
VectorAdd.o: VectorAdd.cu
	$(NVCC) -c VectorAdd.cu
VectorAdd:
	$(NVCC) -o VectorAdd VectorAdd.cu
HelloWorld.o: HelloWorld.cpp
	$(CC) -c HelloWorld.cpp
pvalueiTest:
	$(CC) -c pvalue.cpp
	$(CC) -o pvalue pvalue.o
compile:
	$(CC) -c -DCOMPILE_C *.cpp 
	$(CC) -c -DCOMPILE_C *.c
main:
	$(CC) -o main *.o
clean:
	rm -f *.0
	rm -f *.gch
	rm -f main
all: clean compile main
