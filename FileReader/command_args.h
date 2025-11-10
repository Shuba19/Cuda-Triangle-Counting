#include <string>
#ifndef FILEREADER_COMMAND_ARGS_HPP
#define FILEREADER_COMMAND_ARGS_HPP

struct CommandArgs {
    std::string input_file;
    bool undirect = true;
    bool benchmark = false;
};

CommandArgs parse_command_args(int argc, char** argv);
#endif