PROGRAMNAME = Main
CHATN = simple_triangle_counter
LIB = ./FileReader/FileReader.cpp ./CUDA_Tri_v1/TriangleCounting.cu ./CUDA_Tri_v2/TriangleCounting.cu
T1 = test/t1.graph
T2 = test/t2.graph
T3 = test/t3.graph
T4 = test/t4.graph
T5 = test/t5.graph
T6 = test/t6.graph
T7 = test/t7.graph

# GPU architecture (override with make GPU_ARCH=sm_89)
GPU_ARCH ?= sm_89

.PHONY: all run clean

all: $(PROGRAMNAME)


$(PROGRAMNAME): Main.cu $(LIB)
	nvcc -O2 -std=c++17 -arch=$(GPU_ARCH) -rdc=true  -o $(PROGRAMNAME) Main.cu $(LIB)

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
	
chat:
	echo Chat Program
	
	./$(CHATN) $(T1)
	./$(CHATN) $(T2)
	./$(CHATN) $(T3)
	./$(CHATN) $(T4)
	./$(CHATN) $(T5)
	./$(CHATN) $(T6)

clean:
	rm -f $(PROGRAMNAME)

