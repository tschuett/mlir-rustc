#include "Toml/Toml.h"

#include <fstream>
#include <sstream>
#include <string>

const std::string PATH = "/Users/schuett/Work/aws_ec2_analyzer/Cargo.toml";

using namespace rust_compiler;
using namespace rust_compiler::toml;

int main(int argc, char **argv) {

  std::ifstream t(PATH);
  std::stringstream buffer;
  buffer << t.rdbuf();

  std::string file = buffer.str();

  std::optional<Toml> toml = readToml(file);
  if (toml) {
  }

  return 0;
}
