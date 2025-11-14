# Counting Triangle Project
This is a project made for counting triangles in a graph using parallellization with CUDA.
In this project, I used different approaches to count triangles in a graph.
To run the code, you need to have a machine equipped with a NVIDIA GPU, and cuda toolkit to compile the code.
## Compilation
To compile the code run the ``` make ``` command in the terminal
## Run
To run the code, use the following command in the terminal:
```bash
./out  <arguments>
```
### Arguments
- -i : Input file path (in a METIS style format, look at the  [manual](https://sites.cc.gatech.edu/dimacs10/data/manual.ps) for further information)
- -t : To specify if the user wants to see the time taken to count the triangles
- -d : Specify if the graph is directed, if there is no such flag, the graph is considered undirected
- -h : To display help message
- -help : To display help message
- -b : To run the code in benchmarking mode
- -v : Verbose mode
- -mode : To specify which approach to use for counting triangles. Possible values are:
    - 0: Edge Iterator
    - 1: Node Iterator
    - 2: Tensor Core Matrix Multiplication
### Example
An example of running the code is as follows:
```bash
./out -i input.graph -t -mode 1
```
In this way, the code will take the input graph from the file "input.graph", count the triangles using the Node Iterator approach, and display the time taken to count the triangles.
## TESTING
To run the tests, use the following command in the terminal:
```bash
make run_test
```
This will run the test suite to verify the correctness of the triangle counting implementations.
## Authors
- [Shuba19](https://github.com/Shuba19)