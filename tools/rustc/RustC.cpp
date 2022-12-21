#include "Lexer.h"

#include <string>

#include <fstream>
#include <sstream>

const std::string PATH = "/Users/schuett/Work/aws_ec2_analyzer/src/lib.rs";

int main(int argc, char **argv) {
  std::ifstream t(PATH);
  std::stringstream buffer;
  buffer << t.rdbuf();

  std::string file = buffer.str();

  rust_compiler::lex(file);
}
