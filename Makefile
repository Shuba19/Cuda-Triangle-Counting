PROGRAMNAME = out
SOURCE = Main.cu
LIB1 = ./FileReader/Libs/CUDA_Tri_Edge_Iterator/EdgeIterator.cu
LIB2 = ./FileReader/Libs/CUDA_Tri_Node_Iterator/NodeIterator.cu
LIB3 = ./FileReader/Libs/CUDA_Tri_Tensor_Multi/TensorCalculation.cu
LIB4 = ./FileReader/Libs/CUDA_BitWise/BW_triangle.cu
LIB5 = ./FileReader/Libs/CUDA_Hybrid_Operation/CUDA_Hybrid_Operation.cu
FILER = ./FileReader/FileReader.cu ./FileReader/command_args.cpp
SAMPLE = test/sample.graph
T1 = test/t1.graph
T2 = test/t2.graph
T3 = test/t3.graph
T4 = test/t4.graph
T5 = test/t5.graph
T6 = test/t6.graph
T7 = test/t7.graph

PYTHON = python3 test/FILES/gemini.py 


GPU_ARCH ?= sm_89

.PHONY: all run clean cublas run_cublas

all: $(PROGRAMNAME)


$(PROGRAMNAME): $(SOURCE) $(LIB1) $(LIB2) $(LIB3) $(LIB4) $(LIB5)  $(FILER)
	nvcc -g -G -O2 -std=c++17 -arch=$(GPU_ARCH) -rdc=true  -o $(PROGRAMNAME) $(SOURCE) $(LIB1) $(LIB2) $(LIB3) $(LIB4) $(LIB5) $(FILER)



install: 
	sudo apt-get upgrade && sudo apt-get install build-essential nvidia-common nvidia-cuda-toolkit

run :$(PROGRAMNAME)
	./$(PROGRAMNAME) $(SAMPLE)
run_test :$(PROGRAMNAME)
	./$(PROGRAMNAME) $(T1)
	./$(PROGRAMNAME) $(T2)
	./$(PROGRAMNAME) $(T3)
	./$(PROGRAMNAME) $(T4)
	./$(PROGRAMNAME) $(T5)
	./$(PROGRAMNAME) $(T6)
run_python :$(PROGRAMNAME)
	$(PYTHON) $(T1)
	$(PYTHON) $(T2)
	$(PYTHON) $(T3)
	$(PYTHON) $(T4)
	$(PYTHON) $(T5)
	$(PYTHON) $(T6)
	
clean:
	rm -f $(PROGRAMNAME) $(CUBLAS_PROG)

