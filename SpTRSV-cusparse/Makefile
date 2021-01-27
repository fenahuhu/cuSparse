#compilers
CC=nvcc

#GLOBAL_PARAMETERS
VALUE_TYPE = double

#CUDA_PARAMETERS
NVCC_FLAGS = -O3 -w -m64 -Xptxas -dlcm=cg -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH = /usr/local/cuda-10.2

#includes
INCLUDES = -I$(CUDA_INSTALL_PATH)/include

#libs
#CLANG_LIBS = -stdlib=libstdc++ -lstdc++
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64  -lcudart -lcusparse
LIBS = $(CUDA_LIBS)

#options
#OPTIONS = -std=c99

.PHONY: all clean

.DEFAULT_GOAL := all

all: ./src/main.cu
	$(CC) $(NVCC_FLAGS) ./src/main.cu -o sptrsv $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE) 

clean:
	-rm sptrsv

