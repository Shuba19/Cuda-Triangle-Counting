#include "FileReader.h"
#include <algorithm>
#include <fstream>
#include <iostream>

GraphFR::GraphFR(const std::string fileName)
{
  this->fileName = fileName;
  ReadFile();
}

GraphFR::~GraphFR()
{
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

bool GraphFR::UnweightedRead(std::ifstream &GraphInput)
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
    int edge;
    std::vector<int> v_edge;
    while (iss >> edge)
    {
      v_edge.push_back(edge - 1);
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

bool GraphFR::WeightedRead(std::ifstream &GraphInput)
{
  return true;
}

bool GraphFR::MultiWeightRead(std::ifstream &GraphInput)
{
  std::string line;
  int pos = 0;
  return true;
}

  bool GraphFR::ConstraintReadMW(std::ifstream & GraphInput, int n)
  {
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
    // Skip comment lines at the beginning
    do
    {
      std::getline(GraphInput, line);
    } while (IsMetisComment(line) && !GraphInput.eof());

    std::istringstream iss(line);
    int num;
    while (iss >> num)
    {
      this->args.push_back(num);
    }

    if (this->args.size() != 2 && this->args.size() != 3 && this->args.size() != 4)
    {
      return false;
    }

    this->num_v = this->args.front();
    this->args.pop_front();
    this->num_edge = this->args.front();
    this->args.pop_front();

    bool result = false;
    switch (this->args.size())
    {
    case 0: // only nv and ne in header (unweighted)
      result = UnweightedRead(GraphInput);
      break;
    case 1: // fmt present (weighted)
      if (args.front() == 1)
        result = WeightedRead(GraphInput);
      else if (args.front() == 11)
        result = MultiWeightRead(GraphInput);
      break;
    case 2: // fmt + ncon (constrained)
      args.pop_front();
      result = ConstraintReadMW(GraphInput, args.front());
      break;
    default:
      result = false;
      break;
    }

    GraphInput.close();
    return result;
  }
