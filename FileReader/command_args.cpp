#include "command_args.h"

CommandArgs parse_command_args(int argc, char** argv)
{
    CommandArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--i" && i + 1 < argc) {
            args.input_file = argv[++i];   
        } else if (arg == "--d") {
            args.undirect = false; 
        }
        else if (arg == "--b") {
            args.benchmark = true; 
        }
    }
    return args;
}
