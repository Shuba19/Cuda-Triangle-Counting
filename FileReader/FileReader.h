#ifndef GRAPHFILEREADER

#define GRAPHFILEREADER
#include <string>
#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <sstream>
#include <vector>
class GraphFR{
    int numArgs;
    std::string fileName;
    bool GraphReader(std::ifstream& GraphInput, bool e_weight, bool v_weight, int n_skip);
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