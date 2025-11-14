#include "FileReader/command_args.h"
#include "FileReader/FileReader.h"
int main(int argc, char *argv[])
{
    CommandArgs ca = parse_command_args(argc, argv);
    GraphFR FR(ca);
    FR.ReadFile();
    std::cout << "Count Tri" << FR.CalculateTriangles() << std::endl;

    return 0;
}