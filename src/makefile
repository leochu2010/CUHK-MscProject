CC=/usr/local/gcc481/bin/g++
CFLAGS=-Wall -Wno-sign-compare

LIB_SOURCES=gpu_accel_alg/*.cpp gpu_accel_alg/utils/*.cpp
LIB_CUDA_SOURCES=gpu_accel_alg/*.cu
LIB_INCLUDES=-I /usr/local/cuda/include -I includes

APP_SOURCES=feature_selection/*.cpp
APP_INCLUDES=-I includes -I . -I /usr/local/cuda/include 

NVCC=/usr/local/cuda/bin/nvcc
NVCFLAGS=-ccbin /usr/local/gcc481/bin/g++ -c -arch=sm_20 
OBJS=*.o

GpuAccelAlgLib:
	$(CC) -c -DCOMPILE_C includes/MIToolbox/*.c
	$(CC) -c includes/threadpool/*.cpp $(CFLAGS)
	rm -f threadpool_test.o

GpuAccelAlg:
	$(CC) -c -DCOMPILE_C $(LIB_SOURCES) $(CFLAGS) $(LIB_INCLUDES)
	$(NVCC) -c $(LIB_CUDA_SOURCES) $(NVCFLAGS) $(LIB_INCLUDES) 
	rm -f getgpuinfo.o

FeatureSelection:
	$(CC) -c $(APP_SOURCES) $(CFLAGS) $(APP_INCLUDES)
	$(CC) -o calculate $(OBJS) -L /usr/local/cuda/lib64 -lcudart -lcuda $(CFLAGS) $(APP_INCLUDES)

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
all: clean GpuAccelAlgLib GpuAccelAlg FeatureSelection
