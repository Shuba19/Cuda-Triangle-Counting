#ifndef METISFILEREADER

#define METISFILEREADER
#include <string>
#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <sstream>
#include <list>
#include <vector>
class MetisFR{
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
    MetisFR(const std::string fileName);
    ~MetisFR();
    bool ReadFile();
    int **OutMatrix();
};

#endif