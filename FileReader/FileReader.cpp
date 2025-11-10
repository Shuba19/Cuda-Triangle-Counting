#include "FileReader.h"
#include <algorithm>
#include <fstream>
#include <iostream>

/*********************************
 * Metis Graph first line parameters:
 *
 * - V    : number of vertexes
 * - E    : number of edges
 * - fmt  : can have four values (0 no weight, 1: edges weighted, 10: vertices weighted, 11: edges and vertices weighted)
 * - ncon : number of weight associated to vertices
 /*******************************/

GraphFR::~GraphFR()
{
}

GraphFR::GraphFR(const std::string fileName) : fileName(fileName), num_v(0), num_edge(0)
{
  GraphFR::ReadFile();
}

bool GraphFR::IsMetisComment(const std::string &str)
{
  for (char c : str)
  {
    if (c == '%')
      return true;
  }
  return false;
}

bool GraphFR::GraphReader(std::ifstream &GraphInput, bool e_weight, bool v_weight, int n_skip)
{
  std::string line;
  int pos = 0;
  offsets.push_back(0);
  int sum = 0;
  while (std::getline(GraphInput, line) && pos < num_v)
  {
    if (IsMetisComment(line))
      continue;
    std::istringstream iss(line);
    int garbage;
    if (v_weight)
    {
      iss >> garbage;
      for (int i = 1; i< n_skip; i++)
        iss>> garbage;
    }

    int edge;
    std::vector<int> v_edge;
    while (iss >> edge)
    {
      v_edge.push_back(edge - 1);
      if(e_weight)
        iss >> garbage;
    }
    std::sort(v_edge.begin(), v_edge.end());
    sum = sum + v_edge.size();
    offsets.push_back(sum);
    if (!v_edge.empty())
    {
      for (auto i : v_edge)
        csr.push_back(i);
    }
    pos++;
  }
  return true;
}

bool GraphFR::ReadFile()
{
  std::ifstream GraphInput(this->fileName);
  if (!GraphInput.is_open())
  {
    std::cerr << "Error Opening File, check if the file exists or/and if the name is correct." << std::endl;
    return false;
  }

  std::string line;
  do
  {
    std::getline(GraphInput, line);
  } while (IsMetisComment(line) && !GraphInput.eof());
  if (GraphInput.eof())
    return false;

  std::vector<int> args(4);
  args[0] = 0;
  args[1] = 0;
  args[2] = 0;
  args[3] = 0;
  std::istringstream iss(line);
  int value;
  int p = 0;
  while (iss >> value)
  {
    args[p++] = value;
  }
  bool e_w = (args[2] & 1);
  bool v_w = (args[2] & 2);
  this->num_v = args[0];
  this->num_edge = args[1];
  bool result = GraphReader(GraphInput, e_w, v_w, args[3]);
  GraphInput.close();
  return result;
}
