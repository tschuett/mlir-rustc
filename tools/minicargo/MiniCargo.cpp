#include "Toml/Lexer.h"

#include <fstream>
#include <sstream>
#include <string>

const std::string PATH = "/Users/schuett/Work/aws_ec2_analyzer/Cargo.toml";

int main(int argc, char **argv) {

  std::ifstream t(PATH);
  std::stringstream buffer;
  buffer << t.rdbuf();

  std::string file = buffer.str();

  rust_compiler::toml::lexToml(file);

  return 0;
}
