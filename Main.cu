#include "FileReader/command_args.h"
#include "FileReader/FileReader.h"
int main(int argc, char *argv[])
{
    CommandArgs ca = parse_command_args(argc, argv);
    GraphFR FR(ca);
    FR.ReadFile();
    int triangle_count = 0;
    triangle_count = FR.CalculateTriangles();
    return 0;
}