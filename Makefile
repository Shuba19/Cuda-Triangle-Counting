PROGRAMNAME = out.o
SOURCE = Main.cu
LIB1 = ./Libs/CUDA_Tri_Edge_Iterator/EdgeIterator.cu
LIB2 = ./Libs/CUDA_Tri_Node_Iterator/NodeIterator.cu
LIB3 = ./Libs/CUDA_Tri_Tensor_Multi/TensorCalculation.cu
FILER = ./FileReader/FileReader.cpp 
T1 = test/t1.graph
T2 = test/t2.graph
T3 = test/t3.graph
T4 = test/t4.graph
T5 = test/t5.graph
T6 = test/t6.graph
T7 = test/t7.graph

GPU_ARCH ?= sm_89

.PHONY: all run clean cublas run_cublas

all: $(PROGRAMNAME)


$(PROGRAMNAME): $(SOURCE) $(LIB1) $(LIB2) $(LIB3) $(FILER)
	nvcc -O2 -std=c++17 -arch=$(GPU_ARCH) -rdc=true  -o $(PROGRAMNAME) $(SOURCE) $(LIB1) $(LIB2) $(LIB3) $(FILER)

cublas: $(CUBLAS_PROG)

$(CUBLAS_PROG): CuBlas_Tri.cu $(CUBLAS_LIB)
	nvcc -O2 -std=c++17 -arch=$(GPU_ARCH) -o $(CUBLAS_PROG) CuBlas_Tri.cu $(CUBLAS_LIB) -lcublas

install: 
	sudo apt-get upgrade && sudo apt-get install build-essential nvidia-common nvidia-cuda-toolkit

run: $(PROGRAMNAME)
	echo  -----TEST 1-----
	./$(PROGRAMNAME) $(T1)
	echo  -----TEST 2-----
	./$(PROGRAMNAME) $(T2)
	echo  -----TEST 3-----
	./$(PROGRAMNAME) $(T3)
	echo  -----TEST 4-----
	./$(PROGRAMNAME) $(T4)
	echo  -----TEST 5-----
	./$(PROGRAMNAME) $(T5)
	echo  -----TEST 6-----
	./$(PROGRAMNAME) $(T6)


clean:
	rm -f $(PROGRAMNAME) $(CUBLAS_PROG)

