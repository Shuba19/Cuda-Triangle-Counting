#ifndef GRAPHFILEREADER

#define GRAPHFILEREADER
#include <string>
#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <sstream>
#include <list>
#include <vector>
class GraphFR{
    std::list<int> args;
    int numArgs;
    std::string fileName;
    bool UnweightedRead(std::ifstream& GraphInput);
    bool WeightedRead(std::ifstream& GraphInput);
    bool MultiWeightRead(std::ifstream& GraphInput);
    bool ConstraintReadMW(std::ifstream& GraphInput, int n);
    bool IsMetisComment(const std::string& str);
    public:
    std::vector<int> csr, offsets;
    int num_v, num_edge;
    GraphFR(const std::string fileName);
    ~GraphFR();
    bool ReadFile();
    int **OutMatrix();
};

#endif